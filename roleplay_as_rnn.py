from itertools import imap, islice, chain, izip, tee, groupby
import re
import os
import collections
import random
import argparse
import time
import pprint
import pickle
import sys

delimiters = {
    "python": {
        "start_block": ['"""', "'''"],
        "end_block": ['"""', "'''"],
        "single_line": '#',
    },
    "c-like": {
        "start_block": ['/*'],
        "end_block": ['*/'],
        "single_line": '//',
    },
}

def startswith_any(string, options):
    return any([string.startswith(o) for o in options])

def endswith_any(string, options):
    return any([string.endswith(o) for o in options])

def starts_and_ends(string, start_options, end_options):
    for start in start_options:
        if string.startswith(start) and endswith_any(string[len(start):], end_options):
            return True
    return False

def output(string, verbosity=1):
    """Print gated by verbosity."""
    if args.verbosity >= verbosity:
        print string

def break_line(line):
    """Breaks a line into leading whitespace, body, and trailing whitespace."""
    line = line.rstrip('\r\n')
    return re.match(r"^(\s*)(.*?)(\s*)$", line).groups()

def remove_comments(lines):
    """Generator that removes inline comments and docstrings from lines of code."""
    in_docstring = False
    start_block = delimiters[args.language]["start_block"]
    end_block = delimiters[args.language]["end_block"]
    single_line = delimiters[args.language]["single_line"]
    redaction = '????'
    for line in lines:
        indent, line, _ = break_line(line)
        if in_docstring:
            if endswith_any(line, end_block):
                yield indent + redaction + end_block[0]
                in_docstring = False
            else:
                yield indent + redaction
        else:
            # Check for single-line block
            if starts_and_ends(line, start_block, end_block):
                yield indent + start_block[0] + redaction + end_block[0]
            elif startswith_any(line, start_block):
                yield indent + start_block[0] + redaction
                in_docstring = True
            else:
                i = line.find(single_line)
                if i != -1:
                    yield indent + line[:i] + single_line + ' ' + redaction
                else:
                    yield indent + line

def parse_example_from_queue(q):
    """Given a sliding window of lines (q), build an example from the current window contents.
       Returns None if the current contents do not form a valid example."""
    if len(q) != args.num_context_lines:
        return None
    num_comment_lines = len([x for x in q if x["has_comment"]])
    if num_comment_lines < args.min_comment_lines:
        return None
    groups = groupby(q, key=lambda x: x["has_comment"])
    uncommented_span_lens = [len(list(lines)) for (has_comment, lines) in groups if not has_comment]
    max_len = max(uncommented_span_lens) if uncommented_span_lens else 0
    if max_len > args.max_uncommented_span:
        return None
    if q[-1]["has_comment"]:
        return None
    line = q[-1]["original"]
    text_regions = groupby(enumerate(line),
                           key=lambda x:
                               re.match(args.valid_chars, x[1]) is not None)
    valid_regions = [list(txt) for (valid, txt) in text_regions if valid]
    if not valid_regions:
        return None
    if args.transition_only:
        (split_point, char) = random.choice(valid_regions)[0]
    else:
        (split_point, char) = random.choice(list(flatten(valid_regions)))
    return {
        "context": [item["original"] for item in q][:-1],
        "context_without_comments": [item["without_comments"] for item in q][:-1],
        "line": line[:split_point],
        "next_char": char,
    }        

def parse_examples_from_lines(lines):
    """Generator that parses as many examples as possible given lines of code."""
    skip_lines = 0
    q = collections.deque(maxlen=args.num_context_lines)
    (lines1, lines2) = tee(lines)
    for (line, line_no_comments) in izip(lines1, remove_comments(lines2)):
        line = ''.join(break_line(line)[:2])
        q.append({
            'original': line,
            'without_comments': line_no_comments,
            'has_comment': (line != line_no_comments)
        })
        if skip_lines > 0:
            skip_lines -= 1
            continue
        example = parse_example_from_queue(q)
        if example:
            yield example
            skip_lines = args.min_space_between_examples

def parse_examples_from_file(filename):
    """Generator that parses as many examples as possible from a given file."""
    count = 0
    with open(filename) as f:
        for item in parse_examples_from_lines(f):
            yield item
            count += 1
    output("Parsed %s examples from %s" % (count, filename))

def find_source_files_in_dir(directory):
    """Generator that recursively finds all source files in the given directory."""
    for (dirpath, _, filenames) in os.walk(directory):
        output("Walking directory %s ..." % dirpath)
        for name in filenames:
            _, extension = os.path.splitext(name)
            if extension == args.extension:
                yield os.path.join(dirpath, name)

# Transform an iterable-of-iterables into a flat iterable
flatten = chain.from_iterable

def parse_examples_from_dir(directory):
    """Returns an iterator over all examples parsed from source files in the given directory."""
    source_files = find_source_files_in_dir(directory)
    return flatten(imap(parse_examples_from_file, source_files))

class QuitException(Exception):
    """Raise when user indicates they want to quit."""
    pass

class SkipException(Exception):
    """Raise when user indicates they want to skip an example."""
    pass

def print_help():
    """Print help message showing valid input strings."""
    print "You can enter any of the following:"
    print "A single character matching this regular expression: %s" % args.valid_chars
    print "skip : to skip this example"
    print "quit : to end the program"
    print "help : to display this menu"

def user_input(prompt):
    """Repeatedly prompt user for input until input is valid."""
    answer = raw_input(prompt)
    while len(answer) != 1 or not re.match(args.valid_chars, answer):
        if answer == "quit":
            raise QuitException
        if answer == "skip":
            raise SkipException
        if answer == "help":
            print_help()
        else:
            print "Oops, I didn't understand you"
            print_help()
        answer = raw_input(prompt)
    return answer

def play_example(e):
    """Do user-interaction for the given example, and return the results."""
    print "**********START_EXAMPLE_NO_COMMENTS**********"
    for line in e["context_without_comments"]:
        print line
    print e["line"]
    print re.sub("[^\\t]", " ", e["line"]) + "^"
    print "***********END_EXAMPLE_NO_COMMENTS***********"
    no_comments = user_input("next char? ")
    print "*********START_EXAMPLE_WITH_COMMENTS*********"
    for line in e["context"]:
        print line
    print e["line"]
    print re.sub("[^\\t]", " ", e["line"]) + "^"
    print "**********END_EXAMPLE_WITH_COMMENTS**********"
    with_comments = user_input("next char? ")
    print "Correct answer was %s" % e["next_char"]
    print "No comments %s" % ("CORRECT!" if no_comments == e["next_char"] else "incorrect")
    print "With comments %s" % ("CORRECT!" if with_comments == e["next_char"] else "incorrect")
    print "Press enter to continue"
    raw_input()
    return {
        "no_comments": no_comments,
        "with_comments": with_comments,
    }

def compute_summary(stats):
    """Compute and return summary statistics."""
    no_correct_with_correct = {
        (True, True): 0,
        (True, False): 0,
        (False, True): 0,
        (False, False): 0,
    }
    skipped = 0
    comments_cause_change = 0
    for stat in stats:
        if stat["answers"] is not None:
            with_comment = (stat["answers"]["with_comments"] == stat["example"]["next_char"])
            no_comment = (stat["answers"]["no_comments"] == stat["example"]["next_char"])
            no_correct_with_correct[(no_comment, with_comment)] += 1
            if stat["answers"]["with_comments"] != stat["answers"]["no_comments"]:
                comments_cause_change += 1
        else:
            skipped += 1
    charset_size = len([i for i in range(256) if re.match(args.valid_chars, chr(i))])
    positive_change = no_correct_with_correct[(False, True)]
    negative_change = no_correct_with_correct[(True, False)]
    neutral_change = comments_cause_change - positive_change - negative_change
    return {
        "args": args,
        "results": stats,
        "summary_stats": {
            "correct_without_comments": sum([v for ((no, yes), v) in no_correct_with_correct.iteritems() if no]),
            "correct_with_comments": sum([v for ((no, yes), v) in no_correct_with_correct.iteritems() if yes]),
            "comments_caused_change": comments_cause_change,
            "comments_caused_positive_change": positive_change,
            "comments_caused_negative_change": negative_change,
            "comments_caused_neutral_change": neutral_change,
            "without_correct_cross_with_correct": no_correct_with_correct,
            "skipped": skipped,
            "total_examples": skipped + sum(no_correct_with_correct.itervalues()),
            "charset_size": charset_size,
        },
    }        

def dump_summary(stats):
    """Dump summary stats to disk & console."""
    timestamp = int(time.time())
    directory = os.path.join(args.output_directory, "roleplay_as_rnn_output_%s" % timestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, "pickled_output_dump.p"), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(directory, "human_readable_output_dump.txt"), 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(stats)
    with open(os.path.join(directory, "summary.txt"), 'w') as f:
        def tee_print(string):
            f.write(string + '\n')
            sys.stdout.write(string + '\n')
            sys.stdout.flush()
        num_unskipped_examples = stats["summary_stats"]["total_examples"] - stats["summary_stats"]["skipped"]
        tee_print("Total number of examples played: %s" % num_unskipped_examples)
        if num_unskipped_examples == 0:
            tee_print("Nothing else to report!")
        else:
            tee_print("Accuracy without comments: %.2f%% (%s / %s)" % (
                stats["summary_stats"]["correct_without_comments"] / float(num_unskipped_examples) * 100,
                stats["summary_stats"]["correct_without_comments"],
                num_unskipped_examples))
            tee_print("Accuracy with comments: %.2f%% (%s / %s)" % (
                stats["summary_stats"]["correct_with_comments"] / float(num_unskipped_examples) * 100,
                stats["summary_stats"]["correct_with_comments"],
                num_unskipped_examples))
            tee_print("Expected accuracy with random guessing: %.2f%% (1 / %s (charset size))" % (
                100.0 / stats["summary_stats"]["charset_size"],
                stats["summary_stats"]["charset_size"]))
            tee_print("")
            tee_print("Number of times comments caused changes: %.2f%% (%s / %s)" % (
                stats["summary_stats"]["comments_caused_change"] / float(num_unskipped_examples) * 100,
                stats["summary_stats"]["comments_caused_change"],
                num_unskipped_examples))
            tee_print("Number of times comments caused *positive* changes: %.2f%% (%s / %s)" % (
                stats["summary_stats"]["comments_caused_positive_change"] / float(num_unskipped_examples) * 100,
                stats["summary_stats"]["comments_caused_positive_change"],
                num_unskipped_examples))
            tee_print("Number of times comments caused *negative* changes: %.2f%% (%s / %s)" % (
                stats["summary_stats"]["comments_caused_negative_change"] / float(num_unskipped_examples) * 100,
                stats["summary_stats"]["comments_caused_negative_change"],
                num_unskipped_examples))
            tee_print("Number of times comments caused *neutral* changes: %.2f%% (%s / %s)" % (
                stats["summary_stats"]["comments_caused_neutral_change"] / float(num_unskipped_examples) * 100,
                stats["summary_stats"]["comments_caused_neutral_change"],
                num_unskipped_examples))

def int_at_least(minimum):
    """Returns an argparse-compatible type-converter for integers >= minimum."""
    def parse(string):
        value = int(string)
        if value < minimum:
            msg = "argument (%s) must be greater than %s" % (string, minimum)
            raise argparse.ArgumentTypeError(msg)
        return value
    return parse

def get_args(argv=None):
    """Parse command-line arguments from argv, or from sys.argv if argv is None."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory", required=True,
        help="directory to look for python files")
    parser.add_argument(
        "--output_directory", default=".",
        help="directory to write output results")
    parser.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1],
        help="set verbosity: 0 is quiet, 1 is verbose")
    parser.add_argument(
        "--num_context_lines", type=int_at_least(1), default=15,
        help="number of lines of context to show")
    parser.add_argument(
        "--min_space_between_examples", type=int_at_least(1), default=30,
        help="minimum number of lines required between examples")
    parser.add_argument(
        "--transition_only", type=bool, default=False,
        help="only show examples where the next char is the first in a run of valid chars")
    parser.add_argument(
        "--valid_chars", default=r'[A-Za-z0-9()\[\]:+*/^_.-]',
        help="regular expression matching a valid next char")
    parser.add_argument(
        "--min_comment_lines", type=int_at_least(1), default=4,
        help="minimum number of comment lines required in an example's context")
    parser.add_argument(
        "--max_uncommented_span", type=int_at_least(1), default=None,
        help="maximum allowed number of consecutive lines without a comment")
    parser.add_argument(
        "--num_examples", type=int_at_least(1), default=50,
        help="number of examples to show")
    parser.add_argument(
        "--start_index", type=int_at_least(0), default=0,
        help="number of examples at beginning to skip")
    parser.add_argument(
        "--language", default="python", choices=delimiters.keys(),
        help="what comment style to parse")
    parser.add_argument(
        "--extension", default=".py",
        help="file extension to use when looking for source files")

    args = parser.parse_args(argv)
    if args.max_uncommented_span is None:
        args.max_uncommented_span = args.num_context_lines
        print "max_uncommented_span not set, defaulting to %s" % args.max_uncommented_span
    return args

def main():
    global args
    #args = get_args(["--input_directory", directory])
    args = get_args()

    stats = []
    examples = parse_examples_from_dir(args.input_directory)
    examples = islice(examples, args.start_index, args.start_index + args.num_examples)
    examples = list(examples)
    output("Done parsing; found %s examples" % len(examples))
    output("Shuffling examples ...")
    random.shuffle(examples)
    for e in examples:
        try:
            answers = play_example(e)
            stats.append({
                "example": e,
                "answers": answers
            })
        except SkipException:
            stats.append({
                "example": e,
                "answers": None
            })
        except QuitException:
            break
    dump_summary(compute_summary(stats))
        
if __name__ == "__main__":
    main()
        

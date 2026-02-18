# parse.py

from datetime import datetime
from collections import Counter
import os
import sys
import time
import pandas as pd
import regex as re

RED = "\033[31m"
RESET = "\033[0m"
PINK = "\033[38;2;255;192;203m"


class LogParser:
    def __init__(
        self,
        logname,
        log_format,
        indir="./",
        outdir="./result/",
        threshold=2,
        delimeter=None,
        rex=None,
    ):
        """
        logname     : dataset name (e.g., "Android")
        log_format  : log format string (custom for Android)
        indir       : input log directory
        outdir      : output directory for structured logs
        threshold   : similarity threshold for template split
        delimeter   : delimiter regex list (optional preprocessing)
        rex         : regex list for replacing numbers/addresses/hex etc.
        """
        self.logformat = log_format
        self.path = indir
        self.savePath = outdir
        self.rex = rex if rex is not None else []
        self.df_log = None
        self.logname = logname
        self.threshold = threshold
        self.delimeter = delimeter if delimeter is not None else []

    def parse(self, logName):
        print("Parsing file: " + os.path.join(self.path, logName))
        starttime = datetime.now()
        self.logName = logName

        # 1) Load raw logs into DataFrame
        self.load_data()

        # 2) Extract Content column list and build frequency vectors
        sentences = self.df_log["Content"].tolist()
        group_len, tuple_vector, frequency_vector = self.get_frequecy_vector(
            sentences, self.rex, self.delimeter, self.logname
        )

        # 3) Build word combinations for template generation
        (
            sorted_tuple_vector,
            word_combinations,
            word_combinations_reverse,
        ) = self.tuple_generate(group_len, tuple_vector, frequency_vector)

        # 4) Tree-based parsing using tupletree
        template_set = {}
        for key in group_len.keys():
            Tree = tupletree(
                sorted_tuple_vector[key],
                word_combinations[key],
                word_combinations_reverse[key],
                tuple_vector[key],
                group_len[key],
            )
            root_set_detail_ID, root_set, root_set_detail = Tree.find_root(0)

            # Split parent node
            root_set_detail_ID = Tree.up_split(root_set_detail_ID, root_set)
            # Split child node (by threshold)
            parse_result = Tree.down_split(
                root_set_detail_ID, self.threshold, root_set_detail
            )
            template_set.update(output_result(parse_result))

        endtime = datetime.now()
        print("\nParsing done...")
        print("Time taken   =   " + PINK + str(endtime - starttime) + RESET)

        # 5) Create output folder
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        # 6) Create final CSV with EventId/EventTemplate added
        self.generateresult(template_set, sentences)

    def generateresult(self, template_set, sentences):
        """
        template_set : { (template tuple) : [line index list] }
        sentences    : original Content list
        """
        template_ = len(sentences) * [0]
        EventID = len(sentences) * [0]
        IDnumber = 0
        df_out = []
        for k1 in template_set.keys():
            df_out.append(["E" + str(IDnumber), k1, len(template_set[k1])])
            for i in template_set[k1]:
                template_[i] = " ".join(k1)
                EventID[i] = "E" + str(IDnumber)
            IDnumber += 1

        self.df_log["EventId"] = EventID
        self.df_log["EventTemplate"] = template_
        # Save raw logs + structured results
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logName + "_structured.csv"), index=False
        )

        # Save template list
        df_event = pd.DataFrame(
            df_out, columns=["EventId", "EventTemplate", "Occurrences"]
        )
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    def preprocess(self, line):
        """
        Replace numbers/addresses/hex etc. with <*>
        """
        for currentRex in self.rex:
            line = re.sub(currentRex, "<*>", line)
        return line

    def load_data(self):
        """
        Use custom regex for Android log format
        Otherwise use generate_logformat_regex() + log_to_dataframe()
        """
        # ---------------------------------------------------------------
        # 1) Android dataset (logname == "Android"): custom parsing
        # ---------------------------------------------------------------
        if self.logname.lower() == "android":
            # (1) define columns
            headers = ["Date", "Time", "Level", "Component", "Pid", "Content"]
            # (2) custom regex: "MM-DD HH:MM:SS.mmm L/Component(PID): Content"
            pattern = (
                r"^(?P<Date>\d{2}-\d{2})\s+"
                r"(?P<Time>\d{2}:\d{2}:\d{2}\.\d{3})\s+"
                r"(?P<Level>[VDIWEFS])\/(?P<Component>[\w\.]+)\((?P<Pid>\d+)\):\s+"
                r"(?P<Content>.*)$"
            )
            regex = re.compile(pattern)
            # (3) call Android-specific function
            self.df_log = self.log_to_dataframe_android(
                os.path.join(self.path, self.logName), regex, headers
            )
        else:
            # ----------------------------------------------------------------
            # 2) Other datasets (LogPai style): parse with standard routine
            # ----------------------------------------------------------------
            headers, regex = self.generate_logformat_regex(self.logformat)
            self.df_log = self.log_to_dataframe(
                os.path.join(self.path, self.logName), regex, headers, self.logformat
            )

    def log_to_dataframe_android(self, log_file, regex, headers):
        """
        Android log parsing
        log_file : actual log file path
        regex    : compiled regex
        headers  : ['Date','Time','Level','Component','Pid','Content']
        """
        log_messages = []
        linecount = 0
        with open(log_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    match = regex.match(line)
                    if not match:
                        # skip unmatched lines
                        continue
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception:
                    # ignore regex errors, etc.
                    continue
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", [i + 1 for i in range(linecount)])
        return logdf

    def generate_logformat_regex(self, logformat):
        """
        Following LogPai style format (<Date> <Time> <Pid> <Level> <Component>: <Content>)
        generate regex and return headers
        """
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """
        convert LogPai-style log to DataFrame
        """
        log_messages = []
        linecount = 0
        with open(log_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    match = regex.search(line)
                    if not match:
                        continue
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception:
                    continue
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", [i + 1 for i in range(linecount)])
        return logdf

    def tuple_generate(self, group_len, tuple_vector, frequency_vector):
        """
        Generate word combinations
        """
        sorted_tuple_vector = {}
        word_combinations = {}
        word_combinations_reverse = {}
        for key in group_len.keys():
            for fre in tuple_vector[key]:
                sorted_fre_reverse = sorted(fre, key=lambda tup: tup[0], reverse=True)
                sorted_tuple_vector.setdefault(key, []).append(sorted_fre_reverse)
            for fc in frequency_vector[key]:
                number = Counter(fc)
                result = number.most_common()
                sorted_result = sorted(result, key=lambda tup: tup[1], reverse=True)
                sorted_fre = sorted(result, key=lambda tup: tup[0], reverse=True)
                word_combinations.setdefault(key, []).append(sorted_result)
                word_combinations_reverse.setdefault(key, []).append(sorted_fre)
        return sorted_tuple_vector, word_combinations, word_combinations_reverse

    def get_frequecy_vector(self, sentences, filter, delimiter, dataset):
        """
        Counting each word's frequency in the dataset and convert each log into frequency vector.
        Also print progress bar and ETA.
        Output:
            group_len: logs grouped by length
            tuple_vector: (frequency, word, position) tuple vector
            frequency_vector: pure frequency list per position
        """
        total = len(sentences)
        if total == 0:
            return {}, {}, {}

        print("Building frequency vectors...")

        group_len = {}
        set_dict = {}
        line_id = 0

        # set progress bar
        bar_size = 20
        last_printed = -1
        start_loop = time.time()

        for idx, s in enumerate(sentences):
            # compute progress
            elapsed_loop = time.time() - start_loop
            percent = (idx + 1) / total
            filled = int(percent * bar_size)

            # compute ETA
            if idx > 0:
                time_per_item = elapsed_loop / idx
                remaining = time_per_item * (total - idx - 1)
                eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            else:
                eta = "--:--:--"

            # print progress bar + percent + ETA
            if filled != last_printed and ((idx + 1) == total or (idx + 1) % max(1, total // (bar_size * 2)) == 0):
                bar = "=" * filled + " " * (bar_size - filled)
                sys.stdout.write(f"\rProgress: [{bar}] {percent*100:5.1f}%  ETA: {eta}")
                sys.stdout.flush()
                last_printed = filled

            # parameter substitution (numbers/addresses → <*>)
            for rgex in filter:
                s = re.sub(rgex, "<*>", s)
            # delimiter substitution
            for de in delimiter:
                s = re.sub(de, "", s)

            # dataset-specific preprocessing
            if dataset.lower() == "healthapp":
                s = re.sub(":", ": ", s)
                s = re.sub("=", "= ", s)
                s = re.sub("\|", "| ", s)
            elif dataset.lower() == "android":
                s = re.sub("\(", "( ", s)
                s = re.sub("\)", ") ", s)
                s = re.sub(":", ": ", s)
                s = re.sub("=", "= ", s)
            elif dataset.lower() == "hpc":
                s = re.sub("=", "= ", s)
                s = re.sub("-", "- ", s)
                s = re.sub(":", ": ", s)
            elif dataset.lower() == "bgl":
                s = re.sub("=", "= ", s)
                s = re.sub("\.\.", ".. ", s)
                s = re.sub("\(", "( ", s)
                s = re.sub("\)", ") ", s)
            elif dataset.lower() == "hadoop":
                s = re.sub("_", "_ ", s)
                s = re.sub(":", ": ", s)
                s = re.sub("=", "= ", s)
                s = re.sub("\(", "( ", s)
                s = re.sub("\)", ") ", s)
            elif dataset.lower() == "hdfs":
                s = re.sub(":", ": ", s)
            elif dataset.lower() == "linux":
                s = re.sub("=", "= ", s)
                s = re.sub(":", ": ", s)
            elif dataset.lower() == "spark":
                s = re.sub(":", ": ", s)
            elif dataset.lower() == "thunderbird":
                s = re.sub(":", ": ", s)
                s = re.sub("=", "= ", s)
            elif dataset.lower() == "windows":
                s = re.sub(":", ": ", s)
                s = re.sub("=", "= ", s)
                s = re.sub("\[", "[ ", s)
                s = re.sub("]", "] ", s)
            elif dataset.lower() == "zookeeper":
                s = re.sub(":", ": ", s)
                s = re.sub("=", "= ", s)

            # final whitespace trim then tokenize
            s = re.sub(",", ", ", s)
            tokens = re.sub(" +", " ", s).split(" ")

            # prepend line_id as string to keep index consistency when grouping
            tokens.insert(0, str(line_id))
            length = len(tokens)
            group_len.setdefault(length, []).append(tokens)

            # collect tokens per position i for frequency counts
            for pos, token in enumerate(tokens):
                set_dict.setdefault(str(pos), []).append(token)

            line_id += 1

        # finally show 100% + ETA 00:00:00
        bar = "=" * bar_size
        sys.stdout.write(f"\rProgress: [{bar}] 100.0%  ETA: 00:00:00\n")
        sys.stdout.flush()

        # create freq dict: (position + word) -> freq
        tuple_vector = {}
        frequency_vector = {}

        a = max(group_len.keys())  # max length among groups
        fre_set = {}

        for i in range(a):
            for word in set_dict.get(str(i), []):
                key_word = f"{i} {word}"
                fre_set[key_word] = fre_set.get(key_word, 0) + 1

        # build (freq, token, position) vectors per length group
        for key in group_len.keys():
            for tokens in group_len[key]:
                position = 0
                fre = []
                fre_common = []
                skip_lineid = True
                for word_character in tokens:
                    if skip_lineid:
                        skip_lineid = False
                        continue
                    # use position+1 to avoid KeyError
                    lookup_key = f"{position+1} {word_character}"
                    frequency_word = fre_set.get(lookup_key, 0)
                    tup = (frequency_word, word_character, position)
                    fre.append(tup)
                    fre_common.append(frequency_word)
                    position += 1
                tuple_vector.setdefault(key, []).append(fre)
                frequency_vector.setdefault(key, []).append(fre_common)

        return group_len, tuple_vector, frequency_vector


class tupletree:
    """
    tupletree(sorted_tuple_vector, word_combinations, word_combinations_reverse, tuple_vector, group_len)
    """

    def __init__(
        self,
        sorted_tuple_vector,
        word_combinations,
        word_combinations_reverse,
        tuple_vector,
        group_len,
    ):
        self.sorted_tuple_vector = sorted_tuple_vector
        self.word_combinations = word_combinations
        self.word_combinations_reverse = word_combinations_reverse
        self.tuple_vector = tuple_vector
        self.group_len = group_len

    def find_root(self, threshold_per):
        root_set_detail_ID = {}
        root_set_detail = {}
        root_set = {}
        i = 0
        for fc in self.word_combinations:
            count = self.group_len[i]
            threshold = (max(fc, key=lambda tup: tup[0])[0]) * threshold_per
            m = 0
            for fc_w in fc:
                if fc_w[0] >= threshold:
                    self.sorted_tuple_vector[i].append((int(count[0]), -1, -1))
                    root_set_detail_ID.setdefault(fc_w, []).append(
                        self.sorted_tuple_vector[i]
                    )
                    root_set.setdefault(fc_w, []).append(
                        self.word_combinations_reverse[i]
                    )
                    root_set_detail.setdefault(fc_w, []).append(self.tuple_vector[i])
                    break
                if fc_w[0] >= m:
                    candidate = fc_w
                    m = fc_w[0]
                if fc_w == fc[-1]:
                    self.sorted_tuple_vector[i].append((int(count[0]), -1, -1))
                    root_set_detail_ID.setdefault(candidate, []).append(
                        self.sorted_tuple_vector[i]
                    )
                    root_set.setdefault(candidate, []).append(
                        self.word_combinations_reverse[i]
                    )
                    root_set_detail.setdefault(fc_w, []).append(self.tuple_vector[i])
            i += 1
        return root_set_detail_ID, root_set, root_set_detail

    def up_split(self, root_set_detail, root_set):
        for key in root_set.keys():
            tree_node = root_set[key]
            father_count = []
            for node in tree_node:
                pos = node.index(key)
                for i in range(pos):
                    father_count.append(node[i])
            father_set = set(father_count)
            for father in father_set:
                if father_count.count(father) == key[0]:
                    continue
                else:
                    for i in range(len(root_set_detail[key])):
                        for k in range(len(root_set_detail[key][i])):
                            if father[0] == root_set_detail[key][i][k]:
                                root_set_detail[key][i][k] = (
                                    root_set_detail[key][i][k][0],
                                    "<*>",
                                    root_set_detail[key][i][k][2],
                                )
                    break
        return root_set_detail   # typo fix: root_set_detail → root_set_detail

    def down_split(self, root_set_detail_ID, threshold, root_set_detail):
        for key in root_set_detail_ID.keys():
            thre = threshold
            detail_order = root_set_detail[key]
            m = []
            child = {}
            variable = set()
            variable_set = set()
            m_count = 0
            first_sentence = detail_order[0]
            for det in first_sentence:
                if det[0] != key[0]:
                    m.append(m_count)
                m_count += 1
            for i in m:
                for node in detail_order:
                    if i < len(node):
                        child.setdefault(i, []).append(node[i][1])
            for i in m:
                result = set(child[i])
                freq = len(result)
                if freq >= thre:
                    variable = variable.union(result)
            i = 0
            while i < len(root_set_detail_ID[key]):
                j = 0
                while j < len(root_set_detail_ID[key][i]):
                    tup = root_set_detail_ID[key][i][j]
                    if isinstance(tup, tuple):
                        if tup[1] in variable:
                            root_set_detail_ID[key][i][j] = (tup[0], "<*>", tup[2])
                    j += 1
                i += 1
        return root_set_detail_ID


def output_result(parse_result):
    template_set = {}
    for key in parse_result.keys():
        for pr in parse_result[key]:
            sort = sorted(pr, key=lambda tup: tup[2])
            template = []
            i = 1
            while i < len(sort):
                this = sort[i][1]
                if "<*>" in this:
                    template.append("<*>")
                    i += 1
                    continue
                if exclude_digits(this):
                    template.append("<*>")
                    i += 1
                    continue
                template.append(this)
                i += 1
            template = tuple(template)
            template_set.setdefault(template, []).append(pr[-1][0])
    return template_set


def exclude_digits(string):
    """
    words with high digit ratio are treated as variables (e.g., 1234, 0xABCDEF)
    """
    pattern = r"\d"
    digits = re.findall(pattern, string)
    if len(digits) == 0:
        return False
    return len(digits) / len(string) >= 0.3

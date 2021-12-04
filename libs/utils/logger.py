import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import curses
import sys


class LoggerGroup:
    def __init__(self, title="LoggerGroup"):
        """
        This class will create a dictionary for each variable which contains 3 keys
        - 'f_hist' : A list which store all value steps.
        - 'epchs' : A list which store an average of all steps in each epochs
        - 'each_epch' : Store all step in each epochs (This list will be clear
                        when the flush_epoch_all() has been called)
        """
        self.__dict = {}
        self.title = title

    def add_var(self, *keys):
        for e_k in keys:
            self.__dict[e_k] = {
                'f_hist': [],
                'epchs': [],
                'each_epch': [],
                'sub_each_epch': []
            }

    def get_latest_step(self):
        """
        :return: Return the last variable step of all available variable in this group
        """
        r_obj = {}
        for e_k in self.__dict.keys():
            if len(self.__dict[e_k]['each_epch']) > 0:  # if 'each_epch' steps is not empty return the latest step
                r_obj[e_k] = self.__dict[e_k]['each_epch'][-1]
            elif len(self.__dict[e_k]['epchs']) > 0:  # but if it's no step available, return 'epchs' instead
                r_obj[e_k] = self.__dict[e_k]['epchs'][-1]
            else:  # but if there is no step at all, then return none... let the reporter do the rest
                r_obj[e_k] = None
        return r_obj

    def collect_sub_step(self, key: str, value: float):
        self.__dict[key]['sub_each_epch'].append(value)

    def flush_sub_step_all(self):
        for e_k in self.__dict.keys():
            sub_each_epch = self.__dict[e_k]['sub_each_epch']
            if len(sub_each_epch) > 0:
                self.__dict[e_k]['epchs'] += sub_each_epch
                self.__dict[e_k]['each_epch'].append(sum(sub_each_epch) / len(sub_each_epch))
                self.__dict[e_k]['sub_each_epch'] = []

    def collect_step(self, key: str, value: float):
        self.__dict[key]['each_epch'].append(value)

    def flush_step_all(self):
        for e_k in self.__dict.keys():
            each_epch = self.__dict[e_k]['each_epch']
            if len(each_epch) == 0:  # Skip the key if there is nothing to flush
                continue
            else:
                self.__dict[e_k]['f_hist'] += each_epch
                self.__dict[e_k]['epchs'].append(sum(each_epch) / len(each_epch))
                self.__dict[e_k]['each_epch'] = []

    def plot_all(self):
        for title in [('{} full history'.format(self.title), 'f_hist'),
                      ('{} epoch history'.format(self.title), 'epchs')]:
            plt.title(title[0])
            for e_k in self.__dict.keys():
                arr = self.__dict[e_k][title[1]]
                if len(arr) > 0 :
                    desc = e_k + " mn:{:.2f}, mx:{:.2f}, lst:{:.2f}".format(min(arr), max(arr), arr[-1])
                    plt.plot(arr, label=desc)
            plt.legend()
            plt.show()

    def export_file(self, filename: str):
        if ".lggr" not in filename:
            filename = filename + ".lggr"
        f = open(filename, 'wb')
        obj = {
            'title': self.title,
            'dict': self.__dict
        }
        pickle.dump(obj, f)

    def load_file(self, filename: str):
        f = open(filename, 'rb')
        obj = pickle.load(f)
        self.title = obj['title']
        self.__dict = obj['dict']


class Reporter:
    def __init__(self, *logger_groups):
        self.l_g = logger_groups
        self.line_count = 0  # This variable only use in classic reporter
        self.__init_stdscr()

    def __init_stdscr(self):
        try:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.use_classic_report = False
        except curses.error:
            print("<I> Cannot identify terminal. Using classic reporter")
            self.use_classic_report = True

    def classic_reporter(self, classic_mode: bool):
        if classic_mode:
            self.use_classic_report = True
            self.stop()
        else:
            self.__init_stdscr()

    def report(self, epch=None, b_i=None, b_all=None):
        """

        :param epch: number of epoch that training
        :param b_i: number of batch that in training
        :param b_all: number of all batch in the data loader
        :return:
        """
        if self.use_classic_report:
            report_str = 'Epoch [{:4}|{:3}/{:3}] >> '.format(epch, b_i, b_all)
            for i, e_lg in enumerate(self.l_g):
                report_str += e_lg.title + ' > '
                vars_step = e_lg.get_latest_step()
                for e_v in vars_step.keys():
                    if vars_step[e_v] is not None:
                        report_str += "{}[{:8.4f}],".format(e_v, vars_step[e_v])
                    else:
                        report_str += "{}[N/A],".format(e_v)
                report_str = report_str[0:-1] + ' | '
            sys.stdout.write('\r' + report_str[0:-3])

        else:
            line_idx = 0
            if epch is not None and b_i is not None and b_all is not None:
                self.stdscr.addstr(line_idx, 0, "Epoch [{:4} | {:3}/{:3}]".format(epch, b_i, b_all))
                line_idx += 1

            # Iter on each group
            for i, e_lg in enumerate(self.l_g):
                title = e_lg.title
                self.stdscr.addstr(line_idx, 0, "{}".format(title))
                line_idx += 1
                vars_step = e_lg.get_latest_step()
                var_report = " > "
                for e_v in vars_step.keys():
                    if vars_step[e_v] is not None:
                        var_report += "{}[{:10.4f}], ".format(e_v, vars_step[e_v])
                    else:
                        var_report += "{}[N/A], ".format(e_v)
                self.stdscr.addstr(line_idx, 0, var_report[0:-2])
                line_idx += 1
            self.stdscr.refresh()

    def stop(self):
        if not self.use_classic_report:
            del self.stdscr
            curses.echo()
            curses.nocbreak()
            curses.endwin()

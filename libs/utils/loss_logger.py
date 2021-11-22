import matplotlib.pyplot as plt


class LoggerGroup:
    def __init__(self, title):
        """
        This class will create a dictionary for each variable which contains 3 keys
        - 'f_hist' : A list which store all value steps.
        - 'epchs' : A list which store an average of all steps in each epochs
        - 'each_epch' : Store all step in each epochs (This list will be clear
                        when the flush_epoch_all() has been called)
        """
        self.__dict = {}
        self.title = title
        pass

    def add_var(self, *keys):
        for e_k in keys:
            self.__dict[e_k] = {
                'f_hist': [],
                'epchs': [],
                'each_epch': []
            }

    def collect_step(self, key: str, value: float):
        self.__dict[key]['each_epch'].append(value)

    def flush_epoch_all(self):
        for e_k in self.__dict.keys():
            each_epch = self.__dict[e_k]['each_epch']
            self.__dict[e_k]['f_hist'] += each_epch
            self.__dict[e_k]['epchs'].append(sum(each_epch) / len(each_epch))
            self.__dict[e_k]['each_epch'] = []

    def plot_all(self):
        for title in [('{} full history'.format(self.title), 'f_hist'),
                      ('{} epoch history'.format(self.title), 'epchs')]:
            plt.title(title[0])
            for e_k in self.__dict.keys():
                arr = self.__dict[e_k][title[1]]
                plt.plot(arr, label=e_k)
            plt.legend()
            plt.show()

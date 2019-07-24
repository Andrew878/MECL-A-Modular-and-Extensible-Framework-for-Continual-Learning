import datetime

class RecordKeeper:

    def __init__(self, PATH):
        self.path = PATH

        self.synth_versus_real = {}

        self.delimiter = ','
        self.header_string = ""





    def record_iteration_accuracy(self,test_name, epoch_record):

        if test_name not in self.synth_versus_real.keys():
            self.synth_versus_real[test_name] = [epoch_record]
            self.header_string = self.synth_versus_real[test_name][0].header_string
        else:
            self.synth_versus_real[test_name].append(epoch_record)

    def record_to_file(self, test_name):

        now = datetime.datetime.now()

        date_string = str(now.month)+str(now.day)+".txt"

        string = "\n"

        string += self.header_string+"\n"
        for test_name in self.synth_versus_real:
            for epoch_record in self.synth_versus_real[test_name]:
                string += epoch_record.record_string+"\n"


        print(string)

        with open(date_string, "a") as text_file:
            text_file.write(string)









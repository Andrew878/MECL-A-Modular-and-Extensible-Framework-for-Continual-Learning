import datetime

class RecordKeeper:

    """A data structure that maintains stores Epoch Record objects and records delimited results to a test file"""

    def __init__(self, PATH):
        self.path = PATH
        self.storage_dict = {}
        self.delimiter = '&&'
        self.header_string = ""


    def record_iteration_accuracy(self,test_name, epoch_record):
        """Add a test record to storage. If new test type, add to test then add."""

        if test_name not in self.storage_dict.keys():
            self.storage_dict[test_name] = [epoch_record]
            self.header_string = self.storage_dict[test_name][0].header_string
        else:
            self.storage_dict[test_name].append(epoch_record)

    def record_to_file(self, test_name):

        """Record delimited results to text file in root directory."""

        now = datetime.datetime.now()

        date_string = str(now.month)+str(now.day)+".txt"

        string = "\n"

        string += self.header_string+"\n"
        for test_name in self.storage_dict:
            for epoch_record in self.storage_dict[test_name]:
                string += epoch_record.record_string+"\n"

        print(string)

        with open(date_string, "a") as text_file:
            text_file.write(string)









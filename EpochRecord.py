

class EpochRecord:

    """A data structure that records a training process in delimited format"""

    def __init__(self,test_name, task, epoch_num_VAE, epoch_num_CNN, sample_num, num_classes, overall_ave_recon,overall_acc,  class_labels, recon_per_class, accuracy_per_class, random_image_per_class_list=None):

        self.test_name = test_name
        self.task = task
        self.epoch_num_VAE = str(epoch_num_VAE)
        self.epoch_num_CNN = str(epoch_num_CNN)
        self.sample_num = str(sample_num)
        self.num_classes = str(num_classes)
        self.overall_acc = str(overall_acc)
        self.overall_ave_recon = str(overall_ave_recon)
        self.class_labels = class_labels
        self.accuracy_per_class = str(accuracy_per_class)
        self.recon_per_class = recon_per_class
        self.random_image_per_class_list = random_image_per_class_list
        self.delimeter = "&&"
        self.accuracy_per_class_header = ""

        accuracy_per_class_string = ""
        recon_per_class_string = ""

        for i in range(0,len(accuracy_per_class)):
            accuracy_per_class_string += str(accuracy_per_class[i])+self.delimeter
            recon_per_class_string += str(recon_per_class[i])+self.delimeter
            self.accuracy_per_class_header += self.class_labels[i]+self.delimeter

        self.header_string = test_name + self.delimeter + task+ self.delimeter + str("epoch_num_VAE")+self.delimeter+str("epoch_num_CNN")+self.delimeter + str("sample_num")+ self.delimeter + str("num_classes")+ self.delimeter + str("overall_ave_recon") + self.delimeter + str("overall_acc") +self.delimeter+self.accuracy_per_class_header+self.delimeter+self.accuracy_per_class_header
        self.record_string = test_name + self.delimeter + task+ self.delimeter + self.epoch_num_VAE+self.delimeter+self.epoch_num_CNN + self.delimeter + self.sample_num+ self.delimeter + self.num_classes+ self.delimeter + self.overall_ave_recon + self.delimeter + self.overall_acc +self.delimeter+recon_per_class_string+self.delimeter+accuracy_per_class_string



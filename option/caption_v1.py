import argparse

class CaptionTrainOptions():

    def __init__(self):

        self.initialised = False

    def initialize(self, parser):

        parser.add_argument("-b", "--batch_size", default='50', type=int, help="batch size") # 500
        parser.add_argument("-t", "--train_list", default="datasets/pdbbind_list", help="trainset")
        parser.add_argument("-val", "--val_list", default="datasets/business_list", help="validset")
        # train nci list
        parser.add_argument("-train_nci", "--train_nci_root_path", default="/home/jovyan/transfer_files/fengwei/data/pdbbind_rcomplex1_oddt/", help="path for nci trainset")
        parser.add_argument("-val_nci", "--val_nci_root_path", default="/home/jovyan/transfer_files/data_space/RComplex1/val/newoddts/", help="path for nci validset")



        parser.add_argument("-s", "--save_dir", default="/home/jovyan/project/v9_genmol_exp/pocketbased_genmol/logs", help="save path")
        parser.add_argument("-e", "--epochs", default=1000, type=int, help="batch size")
        parser.add_argument("-l", "--lr", default=0.0001,type=float,help="learning rate")
        parser.add_argument("-d", "--cuda_device", default="0,1", help = "cuda device")

        return parser

    def gather_options(self):

        parser = argparse.ArgumentParser()

        parser = self.initialize(parser)

        return parser.parse_args()

class Args:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 14
        self.lr = 1.0
        self.gamma = 0.7
        self.no_cuda = False
        self.dry_run = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False
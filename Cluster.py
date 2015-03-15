class Cluster(object):
    """docstring for ClassName"""

    def __init__(self, imgs):
        super(Cluster, self).__init__()
        self.imgs = imgs


    def append(self, Cluster):
        self.imgs.append(Cluster.imgs)

    def print_cluster(self):
        print "{"
        for c in self.imgs:
            print c + ", "
        print "}"



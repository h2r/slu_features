import shelve
import cPickle
from environ_vars import SLU_HOME

def make_shelf(fname, outfname):
    db = cPickle.load(open(fname))
    shelf = shelve.open(outfname, protocol=2, writeback=True)
    for key, value in db.iteritems():
        shelf[key] = value
    shelf.close()


def main():
    #make_shelf(SLU_HOME+"/data/flickr/flickr_cache.pck",
    #           SLU_HOME+"/data/flickr/flickr_cache.shelf")
    
    make_shelf(SLU_HOME+"/data/wordnet/wordnet_cache.pck",
               SLU_HOME+"/data/wordnet/wordnet_cache.shelf")
    

if __name__ == "__main__":
    main()





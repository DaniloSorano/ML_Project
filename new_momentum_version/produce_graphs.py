import matplotlib.pyplot as plt

folder = ''
noplot = False
def load_conf_and_score(topology,conf):
        noplot = False
        train_conf = []
        loss = []
        acc = []
        val_loss = []
        val_acc = []
        title = ''
        f = open(folder+'Net_'+str(topology)+'_'+str(conf)+'.score','rb')
        
        print folder+'Net_'+str(topology)+'_'+str(conf)+'.score'
        i = 0
        for line in f:
            line = line.strip()
            if i==0:
                atts = line.split(' ')
                print atts
                for a in atts[2:]:
                    title = title + a +' '
                if atts[1]!='1.0':
                    noplot = True
                    break
                else:
                    train_conf = atts[2:]
            if i>1:
                l,a = line.split('\t')
                
                loss.append(l)
                acc.append(a)
            i = i+1
        f.close()
        return title,loss,acc

def plot_stats(title,name,loss,acc,val_loss=[],val_acc=[],):
        plt.axis('equal')
        pltname = 'Net_' + name
        c = len(loss)
        plt.subplot(211)
        plt.title(title)
        plt.plot(range(0,c),acc,'k')
        plt.xlabel('ephocs')
        plt.ylabel('Acc')
        plt.subplot(212)
        plt.plot(range(0,c),loss,'k')
        plt.xlabel('ephocs')
        plt.ylabel('Loss')
        plt.savefig(folder+pltname)
        plt.clf()

for i in range(0,6):
    for j in range(0,150):
        print 'plot'+str(i)+'_'+str(j)
        confs,l,a = load_conf_and_score(i,j)
        if l!=[]:
            plot_stats(confs,str(i)+'_'+str(j),l,a)

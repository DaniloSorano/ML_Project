import matplotlib.pyplot as plt

folder = 'exp/'
noplot = False
name = 'Net_0_try_2_20.score'
def load_conf_and_score(topology,tries,conf):
        noplot = False
        train_conf = []
        loss = []
        acc = []
        val_loss = []
        val_acc = []
        title = ''
        print folder+'Net_'+str(topology)+'_try_'+str(tries)+'_'+str(conf)+'.score'
        f = open(folder+'Net_'+str(topology)+'_try_'+str(tries)+'_'+str(conf)   +'.score','rb')
        i = 0
        for line in f:
            line = line.strip()
            if i==0:
                atts = line.split(' ')
                print atts
                for a in atts[2:]:
                    title = title + a +' '
                #if atts[1]!='1.0':
                    #noplot = True
                    #break
                #else:
                train_conf = atts[2:]
            if i>1:
                metrics = line.split('\t')
                
                loss.append(metrics[0])
                acc.append(metrics[1])
                val_loss.append(metrics[2])
                val_acc.append(metrics[3])
            i = i+1
        f.close()
        return title,loss,acc,val_loss,val_acc

def plot_stats(title,name,loss,acc,val_loss=[],val_acc=[],):
        plt.axis('equal')
        pltname = 'Net_' + name
        c = len(loss)
        plt.subplot(211)
        plt.title(title)
        plt.plot(range(0,c),acc,'k',range(0,c),val_acc,'r--')
        plt.xlabel('ephocs')
        plt.ylabel('Acc')
        plt.subplot(212)
        plt.plot(range(0,c),loss,'k',range(0,c),val_loss,'r--')
        plt.xlabel('ephocs')
        plt.ylabel('Loss')
        plt.savefig(folder+pltname)
        plt.clf()

for topology in range(0,3):
    for conf in range(0,45):
        for tries in range(0,5):    
            confs,l,a,v_l,v_a = load_conf_and_score(topology,tries,conf)
            if l!=[]:
                plot_stats(confs,str(topology)+'_'+str(tries),l,a,v_l,v_a)
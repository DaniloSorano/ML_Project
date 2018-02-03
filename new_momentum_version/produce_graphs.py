import numpy as np
import matplotlib.pyplot as plt

folder = 'exp_cup/'
noplot = False
def refactor_data(topology,tries,conf):
    #print folder+'Net_'+str(topology)+'_try_'+str(tries)+'_'+str(conf)+'.score'
    f = open(folder+'Net_'+str(topology)+'_try_'+str(tries)+'_'+str(conf)+'.score', 'rb')
    i = 0
    to_write = []
    for line in f:
        line = line.strip()
        if i == 0:
            atts = line.split(' ')
            old_atts = list(atts)
            for i,el in enumerate(atts):
                if el == 'batch_size':
                    old_atts[i]='batch_size:'
                    old_atts[i+1]=old_atts[i+2]
                    old_atts=old_atts[:(i+1)]+old_atts[i+2:]
            #print atts
            if old_atts!=atts:
                print atts
                s=''
                for el in old_atts:
                    s=s+el+' '
                print s
                to_write.append(s)
            else:
                to_write.append(line)
        else:
            to_write.append(line)
        i = i+1
    f.close()
    f = open(folder+'Net_'+str(topology)+'_try_'+str(tries)+'_'+str(conf)+'.score', 'wb')
    i = 0
    for line in to_write:
        f.write(line+'\r\n')
    f.close()
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
                test_acc= float(atts[1])
                for a in atts[2:]:
                    title = title + a +' '
                #if atts[1]!='1.0':
                    #noplot = True
                    #break
                #else:
                train_conf = atts[2:]
            if i>1:
                metrics = line.split('\t')
                metrics = [float(el) for el in metrics]
                loss.append(metrics[0])
                acc.append(metrics[1])
                val_loss.append(metrics[2])
                val_acc.append(metrics[3])
            i = i+1
        f.close()
        return title,test_acc,loss,acc,val_loss,val_acc
def load_conf_and_score_blind(topology,tries,conf):
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
                test_loss= float(atts[1])
                print atts
                if atts[9]=='minibatch':
                    atts[9]='mb'
                    atts[10]='b_size'
                for a in atts[2:]:
                    title = title + a +' '

                #if atts[1]!='1.0':
                    #noplot = True
                    #break
                #else:
                train_conf = atts[2:]
            if i>1:
                metrics = line.split('\t')
                metrics = [float(el) for el in metrics if el!='']
                loss.append(metrics[0])
                val_loss.append(metrics[1])
            i = i+1
        f.close()
        return title,test_loss,loss,val_loss

def plot_stats_reg(name,loss,val_loss=[],color = 'light',title=''):
    if color!='light':
        c1, c2 = 'k', 'r'
    else:
        c1, c2 = 'lightgray', 'peachpuff'
    pltname = 'Net_' + name
    if title != '':
        plt.title(title, fontsize=20)
    c = len(loss)
    plt.plot(range(0,c),loss,linestyle='-',color=c1,label=('Loss average'if color!='light' else ''))
    plt.plot(range(0,c),val_loss,linestyle='--',color=c2,label=('Val_loss average'if color!='light' else '')) 
    
    #plt.set_ylim(.0,0.3)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    if color!='light':
        plt.legend(fontsize=15,loc=1)
    

def plot_stats(name,loss,acc,val_loss=[],val_acc=[],color = 'light',title=''):
    if color!='light':
        c1, c2 = 'k', 'r'
    else:
        c1, c2 = 'lightgray', 'peachpuff'
    pltname = 'Net_' + name
    c = len(loss)
    ax1 = plt.subplot(211)
    if title != '':
        plt.title(title, fontsize=20)
    plt.plot(range(0, c), acc, linestyle='-', color=c1, label=('Acc average'if color != 'light' else ''))
    plt.plot(range(0, c), val_acc, linestyle='--', color=c2, label=('Test_acc average'if color != 'light' else ''))
    ax1.set_ylim(0.2,1.05)
    #x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,0.2,1.05))
    plt.ylabel('Acc')
    if color!='light':
        plt.legend(fontsize=15,loc=4)
    ax2 = plt.subplot(212)
    plt.plot(range(0,c),loss,linestyle='-',color=c1,label=('Loss average'if color!='light' else ''))
    plt.plot(range(0,c),val_loss,linestyle='--',color=c2,label=('Val_loss average'if color!='light' else ''))   
    #x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,0.4,-0.05))
    ax2.set_ylim(.0,0.3)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    if color!='light':
        plt.legend(fontsize=15,loc=1)
    #plt.savefig(folder+str(pltname))
    #plt.clf()



plt.figure(figsize=(10,10))
n_tries = 5
for topology in [2]:
    for conf in range(0,(58)):# if topology==1 else 9)):
        confs,test_loss,l,v_l = load_conf_and_score_blind(topology,0,conf)
        test_loss_avg = np.array(test_loss)
        l_avg = np.array(l)
        v_l_avg = np.array(v_l)
        plot_stats_reg(str(topology)+'_'+str(0),l,v_l)
        for tries in range(1,n_tries):
            confs,test_loss,l,v_l = load_conf_and_score_blind(topology,tries,conf)
            test_loss_avg = test_loss_avg + np.array(test_loss)
            l_avg = l_avg + np.array(l)
            v_l_avg = v_l_avg + np.array(v_l)
            plot_stats_reg(str(topology)+'_'+str(tries),l,v_l)
        test_loss_avg = test_loss_avg /float(n_tries)
        l_avg = l_avg/float(n_tries)
        v_l_avg = v_l_avg/float(n_tries)
        confs = 'Test loss average: '+ str(test_loss_avg) +'\n' + confs
        plot_stats_reg(str(topology)+'_'+str(tries),l_avg,v_l_avg,color='dark',title=confs)
        plt.savefig(folder+str(topology)+'_'+str(conf)+'_Test-'+str(test_loss_avg).replace('.',','))
        plt.clf()
        print 'loss',l_avg[-1],'val loss avg',v_l_avg[-1]

'''
plt.figure(figsize=(10,10))
n_tries = 5
for topology in [0,1,2]:
    for conf in range(0,(45)):# if topology==1 else 9)):
        confs,test_acc,l,a,v_l,v_a = load_conf_and_score(topology,0,conf)
        test_acc_avg = np.array(test_acc)
        l_avg = np.array(l)
        a_avg = np.array(a)
        v_l_avg = np.array(v_l)
        v_a_avg = np.array(v_a)
        if l!=[]:
            plot_stats(str(topology)+'_'+str(0),l,a,v_l,v_a)
        for tries in range(1,n_tries):
            confs,test_acc,l,a,v_l,v_a = load_conf_and_score(topology,tries,conf)
            test_acc_avg = test_acc_avg + np.array(test_acc)
            l_avg = l_avg + np.array(l)
            a_avg = a_avg + np.array(a)
            v_l_avg = v_l_avg + np.array(v_l)
            v_a_avg = v_a_avg + np.array(v_a)
            if l!=[]:
                plot_stats(str(topology)+'_'+str(tries),l,a,v_l,v_a)
        test_acc_avg = test_acc_avg /float(n_tries)
        l_avg = l_avg/float(n_tries)
        a_avg = a_avg/float(n_tries)
        v_l_avg = v_l_avg/float(n_tries)
        v_a_avg = v_a_avg/float(n_tries)
        confs = 'Test Acc average: '+ str(test_acc_avg) +'\n' + confs
        plot_stats(str(topology)+'_'+str(tries),l_avg,a_avg,v_l_avg,v_a_avg,color='dark',title=confs)
        plt.savefig(folder+str(topology)+'_'+str(conf)+'_Test-'+str(test_acc_avg).replace('.',','))
        plt.clf()
        print 'acc',a_avg[-1],'loss',l_avg[-1],'test loss avg',v_l_avg[-1]'''
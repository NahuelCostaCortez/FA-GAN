'''
Nahuel.- With this class different types of arrhythmia can be generated  
'''

import sys
import numpy as np
from math import ceil

def split(samples, proportions, normalise=False, scale=False, labels=None, random_seed=None):
    """
    Return train/validation/test split.
    """
    if random_seed != None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    assert np.sum(proportions) == 1
    n_total = samples.shape[0]
    n_train = ceil(n_total*proportions[0])
    n_test = ceil(n_total*proportions[2])
    n_vali = n_total - (n_train + n_test)
    # permutation to shuffle the samples
    shuff = np.random.permutation(n_total)
    train_indices = shuff[:n_train]
    vali_indices = shuff[n_train:(n_train + n_vali)]
    test_indices = shuff[(n_train + n_vali):]
    # TODO when we want to scale we can just return the indices
    assert len(set(train_indices).intersection(vali_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(vali_indices).intersection(test_indices)) == 0
    # split up the samples
    train = samples[train_indices]
    vali = samples[vali_indices]
    test = samples[test_indices]
    # apply the same normalisation scheme to all parts of the split
    if normalise:
        if scale: raise ValueError(normalise, scale)        # mutually exclusive
        train, vali, test = normalise_data(train, vali, test)
    elif scale:
        train, vali, test = scale_data(train, vali, test)
    if labels is None:
        return train, vali, test
    else:
        print('Splitting labels...')
        if type(labels) == np.ndarray:
            train_labels = labels[train_indices]
            vali_labels = labels[vali_indices]
            test_labels = labels[test_indices]
            labels_split = [train_labels, vali_labels, test_labels]
        elif type(labels) == dict:
            # more than one set of labels!  (weird case)
            labels_split = dict()
            for (label_name, label_set) in labels.items():
                train_labels = label_set[train_indices]
                vali_labels = label_set[vali_indices]
                test_labels = label_set[test_indices]
                labels_split[label_name] = [train_labels, vali_labels, test_labels]
        else:
            raise ValueError(type(labels))
        return train, vali, test, labels_split

def normalise_data(train, vali, test, low=-1, high=1):
    min_val = np.nanmin(np.vstack([train, vali]), axis=(0, 1))
    max_val = np.nanmax(np.vstack([train, vali]), axis=(0, 1))

    normalised_train = (train - min_val)/(max_val - min_val)
    normalised_train = (high - low)*normalised_train + low

    normalised_vali = (vali - min_val)/(max_val - min_val)
    normalised_vali = (high - low)*normalised_vali + low

    normalised_test = (test - min_val)/(max_val - min_val)
    normalised_test = (high - low)*normalised_test + low
    return normalised_train, normalised_vali, normalised_test

def markov_examples(num_examples, betaNA, alfa):
    # Simulación del electrocardiograma de un paciente

    # Todos los tiempos en días

    lambdaGA = 1.0/(3.0/24) # 3 horas
    lambdaA0 = 1.0/2.0 # 2 días
    pAN = 0.75
    pAG = 1 - pAN
    lambdaNA = 1.0/betaNA # 6 meses

    alpha = alfa # Paso de paroxistica a permanente

    ENORMAL = 0
    EGAP = 1
    EARRITMIA = 2
    nombreEstado = [ "NORMAL","ARRITMIA","GAP"]
    samples = []
    
    for i in range(num_examples):
        estado = ENORMAL
        tiemposimulacionmax = 365*10
        dia = 0.0
        eventos = []
        tiempos = [0]
        listaestados = [estado]
        listaestadoscontiempo = {dia: estado}
        for s in np.arange(1000):
            if estado == ENORMAL:
                betaNA = 1.0/lambdaNA
                espera = np.random.exponential(betaNA)
                siguienteEstado = EARRITMIA
            elif estado == EARRITMIA:
                lA = lambdaA0 * (alpha**dia)
                mu2 = lA * pAN
                l1 = lA * pAG
                betaAN = 1.0/mu2
                betaAG = 1.0/l1
                esperaAN = np.random.exponential(betaAN)
                esperaAG = np.random.exponential(betaAG)
                if esperaAN<esperaAG:
                    siguienteEstado = ENORMAL
                    espera = esperaAN
                else:
                    siguienteEstado = EGAP
                    espera = esperaAG
            else:
                betaGA = 1.0/lambdaGA
                espera = np.random.exponential(betaGA)
                siguienteEstado = EARRITMIA
            dia = dia + espera
            estado = siguienteEstado
            #print("Cambio de estado en dia:",dia)
            #print("Nuevo estado:",nombreEstado[siguienteEstado])
            if siguienteEstado==EARRITMIA:
                eventos = eventos + [dia]
            tiempos = tiempos + [dia]
            listaestados = listaestados + [siguienteEstado]
            listaestadoscontiempo[dia]=siguienteEstado
            if dia>tiemposimulacionmax:
                break
        dias=[]
        for dia in range(365*10): 
            dias.append(dia)
        porcentaje = sample=np.zeros(len(dias))
        # Codificación de porcentaje de tiempo en arritmia por dia
        keys=list(listaestadoscontiempo.keys())
        for i in dias:
            for dia in range(len(keys)):
                # Si estoy en el último día en el que se marca un evento, no tengo para comparar con el siguiente día,
                # entonces lo que se hace es marcar ese día con el último cambio de estado registrado
                if dia == len(keys):
                    if i > int(keys[dia]):
                        if listaestadoscontiempo[keys[dia]] == 2:
                            porcentaje[i]=1
                # Si no es el último día en el que hubo cambio de estado miro a ver si en el dia en el que estoy
                # comprobando se produjo algún cambio de estado
                elif i == int(keys[dia]):
                    # Si estuvo a 2 en ese día
                    if listaestadoscontiempo[keys[dia]] == 2:
                        # Calcular el porcentaje de tiempo en el que estuvo a 2
                        horas = abs(keys[dia]) - abs(int(keys[dia]))
                        # Si el siguiente cambio de estado sucede en el mismo dia hay que calcular el porcentaje de tiempo que estuvo
                        # en arritmia ese dia
                        if int(keys[dia+1]) == int(keys[dia]):
                            horas2 = abs(keys[dia+1]) - abs(int(keys[dia+1]))
                            if porcentaje[i]>0:
                                porcentaje[i]+=((horas2-horas)/1)
                            else:
                                porcentaje[i]=(horas2-horas)/1
                        # Si el siguiente cambio de estado sucede en otro dia, significa que el 100€ del tiempo
                        # de este dia estuvo en arritmia, a no ser que ya hubiera estado en arritmia ese dia
                        else:
                            if porcentaje[i]>0:
                                horas = abs(keys[dia]) - abs(int(keys[dia]))
                                porcentaje[i]+=1-horas
                            else:
                                porcentaje[i]=1-(abs(keys[dia]) - abs(int(keys[dia])))
                # Si no hay ningun cambio de estado en este dia puede ser que esté en arritmia o no, 
                # se comprueba mirando el último cambio de estado (seguirá en ese)
                else:
                    if i > int(keys[dia]) and i< int(keys[dia+1]):
                        # Si el último estado marcado es arritmia, sigo en arritmia
                        if listaestadoscontiempo[keys[dia]] == 2:
                            porcentaje[i]=1
                        # Si el último estado marcado no fue arritmia, estoy en estado normal, el porcentaje es 0
        # Después se suavizan los valores
        n_points = len(dias)
        x_vals = np.arange(n_points)
        y_vals=porcentaje
        def fwhm2sigma(fwhm):
             return fwhm / np.sqrt(8 * np.log(2))
        FWHM = 250
        sigma = fwhm2sigma(FWHM)
        smoothed_vals = np.zeros(y_vals.shape)
        for x_position in x_vals:
             kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
             kernel = kernel / sum(kernel)
             smoothed_vals[x_position] = sum(y_vals * kernel)
        # Por último se escala la gráfica a otra con mucho menos puntos
        values = np.zeros(int(len(smoothed_vals)/50))
        cont = 0
        pos = 0
        i = 0
        while i < len(smoothed_vals):
            while cont<50:
                values[pos] += smoothed_vals[i]
                i += 1
                cont += 1
            if cont==50:
                cont = 0
                values[pos] = values[pos]/50
                pos += 1
        samples.append(values)	
    samples = np.array(samples)
    samples=samples.reshape(samples.shape[0],samples.shape[1],1)
    return samples

samples = markov_examples(10, float(sys.argv[1]), float(sys.argv[2]))
#samples = markov_examples(14000, float(sys.argv[1]), float(sys.argv[2]))
train, vali, test = split(samples, [0.6, 0.2, 0.2], normalise=True)
samples = dict()
samples['train'], samples['vali'], samples['test'] = train, vali, test

data_path = './markov_'+str(sys.argv[2][2:])+'na'+str(sys.argv[1])+'.npy'
np.save(data_path, {'samples': samples})
print('Saved training data to', data_path)

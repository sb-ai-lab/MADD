from ic_50_compute import call_for_ic50
import pickle
import time


# run generation by pretrained model
# show elapsed time
# show number of uniq molecules
# show % molecules with low IC50
if __name__ == '__main__':
    with open('/projects/CoScientist/ChemCoScientist/generative_models/GAN/gan_lstm_refactoring/weights/v4_gan_mol_124_0.0003_8k.pkl', "rb") as f:
        gan_mol = pickle.load(f)

    gan_mol.eval()
    
    start_time = time.time()
    samples = gan_mol.generate_n(10000)
    print(samples)
    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print('Elapsed time for 10000 mols: ', elapsed_time)

    # samples_no_dub = list(dict.fromkeys(samples))
    # print('Num molecules without dublicates: ', len(samples_no_dub))

    # _, ic_50 = call_for_ic50(samples)
    # ic50_low = len([i for i in list(ic_50.values())[0] if i <= 6.0]) / 100
    # print(f'Low IC50 is {ic50_low}%')
    
    # ev = gan_mol.evaluate_n(1000)
    # print(ev)


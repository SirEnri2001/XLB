from src.lattice import LatticeD2Q9
from src.utils import *
from termcolor import colored
from tqdm import tqdm
import model_settings

def instantiate_simulator(diam, with_cnn_correction=False, transfer_output=True, quiet=False):
    precision = 'f64/f64'
    prescribed_vel = 0.006
    lattice = LatticeD2Q9(precision)
    nx = int(22 * diam)
    ny = int(4.1 * diam)
    Re = 100.0
    visc = prescribed_vel * diam / Re
    omega = 1.0 / (3. * visc + 0.5)
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'diam': diam,
        'transfer_output': transfer_output,
        'quiet': quiet,
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    if with_cnn_correction:
        return model_settings.CylinderForce(**kwargs)
    else:
        return model_settings.Cylinder(**kwargs)

def generate_sim_dataset(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None):
    nx = int(22 * diam)
    ny = int(4.1 * diam)
    prescribed_vel = 0.006
    tc = prescribed_vel / diam
    if ts_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear. Recommend value is {}".format(
            int(100 // tc)), "red"))
    sim = instantiate_simulator(diam)
    generated_data = sim.run(ts_end, ts_start, output_offset, output_stride, init_f=init_f)
    print("sim completed, data postprocessing ...")
    generated_data['timestep'] = np.array(generated_data['timestep'])
    generated_data['f_poststreaming'] = np.stack(generated_data['f_poststreaming'], axis=0)
    generated_data['rho'] = np.stack(generated_data['rho'], axis=0)
    generated_data['u'] = np.stack(generated_data['u'], axis=0)
    return generated_data

def batch_generator_generate_sim_dataset(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None):
    nx = int(22 * diam)
    ny = int(4.1 * diam)
    prescribed_vel = 0.006
    tc = prescribed_vel / diam
    if ts_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear. Recommend value is {}".format(
            int(100 // tc)), "red"))
    sim = instantiate_simulator(diam)
    return sim.run_batch_generator(ts_end, ts_start, output_offset, output_stride, init_f=init_f)

def read_data(total_batch, file_name="batched_ref_data"):
    res_data = {
        'timestep':[],
        'f_poststreaming':[],
        'u':[]
    }
    for i in tqdm(range(total_batch)):
        loaded_data = np.load('./data/{}_{}.npz'.format(file_name, i))
        res_data['timestep'].append(loaded_data['timestep'])
        res_data['f_poststreaming'].append(loaded_data['f_poststreaming'])
        res_data['u'].append(loaded_data['u'])
    print("concatenating ... ")
    res_data['timestep'] = np.concatenate(res_data['timestep'], axis=0)
    res_data['f_poststreaming'] = np.concatenate(res_data['f_poststreaming'], axis=0)
    res_data['u'] = np.concatenate(res_data['u'], axis=0)
    return res_data

def read_data_and_downsample(total_batch, factor=4, file_name='batched_ref_data'):
    def batch_read_file(loaded_data):
        resized_shape = (
        loaded_data['u'].shape[0], round(loaded_data['u'].shape[1] / factor), round(loaded_data['u'].shape[2] / factor))
        return loaded_data['timestep'], resample_field(loaded_data['u'], factor, shape=(
        resized_shape[0], resized_shape[1], resized_shape[2], loaded_data['u'].shape[3])), resample_field(loaded_data['f_poststreaming'], factor, shape=(
        resized_shape[0], resized_shape[1], resized_shape[2], loaded_data['f_poststreaming'].shape[3]))

    def batch_read(total_batch):
        # for i in tqdm(range(total_batch)):
        #     yield np.load('./data/batched_ref_data_{}.npz'.format(i))
        from concurrent.futures import ThreadPoolExecutor
        tasks = []
        with ThreadPoolExecutor(max_workers=24) as executor:
            for i in range(total_batch):
                loaded_data = np.load("./data/{}_{}.npz".format(file_name, i))
                tasks.append(executor.submit(batch_read_file, loaded_data))
            for task in tqdm(tasks):
                yield task.result()
    res_data = {
        'timestep':[],
        'u':[],
        'f_poststreaming':[]
    }
    for ts, u, f in batch_read(total_batch):
        res_data['timestep'].append(ts)
        res_data['u'].append(u)
        res_data['f_poststreaming'].append(f)
    print("concatenating ... ")
    res_data['timestep'] = np.concatenate(res_data['timestep'], axis=0)
    res_data['u'] = np.concatenate(res_data['u'], axis=0)
    res_data['f_poststreaming'] = np.concatenate(res_data['f_poststreaming'], axis=0)
    return res_data

def generate_sim_dataset_and_save(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None, output_file_name="batched_ref_data"):
    def process_batched_data(batched_data, seq_num, pbar):
        batched_data['timestep'] = np.array(batched_data['timestep'])
        batched_data['u'] = np.stack(batched_data['u'], axis=0)
        batched_data['rho'] = np.stack(batched_data['rho'], axis=0)
        np.savez_compressed("./data/{}_{}.npz".format(output_file_name, seq_num), timestep=batched_data['timestep'], u=batched_data['u'], f_poststreaming=batched_data['f_poststreaming'])
        pbar.set_description("Seq {} saved".format(seq_num))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=24) as executor:
        seq = 0
        tasks = []
        generator = batch_generator_generate_sim_dataset(diam, ts_start, ts_end, output_offset, output_stride, init_f)
        pbar = next(generator)
        for batched_data in generator:
            tasks.append(executor.submit(process_batched_data, batched_data, seq, pbar))
            seq += 1
            while len(tasks) >= 24:
                for task in tasks:
                    if task.done():
                        tasks.remove(task)
        print("Finishing batch tasks ... ")
        for task in tqdm(tasks):
            task.result()

def generate_train_test_dataset():
    generate_sim_dataset_and_save(40, 0, 240000, output_offset=200000, output_stride=8000, init_f=None,
                                  output_file_name='init_frames')
    init_data_frames = read_data(1, file_name='init_frames')
    train_init_f = init_data_frames['f_poststreaming'][0]
    test_init_f = init_data_frames['f_poststreaming'][1]
    generate_sim_dataset_and_save(40, 0, 4000, output_offset=0, output_stride=8, init_f=train_init_f, output_file_name='train_ref_data')
    generate_sim_dataset_and_save(40, 0, 4000, output_offset=0, output_stride=8, init_f=test_init_f, output_file_name='test_ref_data')

def read_train_test_dataset(batch_num=51):
    train_ref_data = read_data_and_downsample(batch_num, 8, file_name='train_ref_data')
    test_ref_data = read_data_and_downsample(batch_num, 8, file_name='test_ref_data')
    return train_ref_data, test_ref_data
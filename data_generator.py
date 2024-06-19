from src.lattice import LatticeD2Q9
from src.utils import *
from termcolor import colored
from tqdm import tqdm
import model_settings
def generate_sim_dataset(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None):
    precision = 'f64/f64'
    # diam_list = [10, 20, 30, 40, 60, 80]
    scale_factor = 2
    prescribed_vel = 0.003 * scale_factor
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
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    # characteristic time
    tc = prescribed_vel / diam
    if ts_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear. Recommend value is {}".format(
            int(100 // tc)), "red"))
    sim = model_settings.Cylinder(**kwargs)
    if init_f is not None:
        loaded_data = np.load('./data/init_frame.npz')
        init_f = resample_field(loaded_data['f'], shape=(nx, ny, loaded_data['f'].shape[2]))
        generated_data = sim.run(ts_end, ts_start, output_offset, output_stride, init_f=init_f)
    else:
        generated_data = sim.run(ts_end, ts_start, output_offset, output_stride)
    print("sim completed, data postprocessing ...")
    generated_data['timestep'] = np.array(generated_data['timestep'])
    generated_data['f_poststreaming'] = np.stack(generated_data['f_poststreaming'], axis=0)
    generated_data['rho'] = np.stack(generated_data['rho'], axis=0)
    generated_data['u'] = np.stack(generated_data['u'], axis=0)
    return generated_data

def batch_generator_generate_sim_dataset(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None):
    precision = 'f64/f64'
    # diam_list = [10, 20, 30, 40, 60, 80]
    scale_factor = 2
    prescribed_vel = 0.003 * scale_factor
    lattice = LatticeD2Q9(precision)

    nx = int(22 * diam)
    ny = int(4.1 * diam)

    Re = 100.0
    visc = prescribed_vel * 40 / Re
    omega = 1.0 / (3. * visc + 0.5)
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'diam': diam,
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    # characteristic time
    tc = prescribed_vel / diam
    if ts_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear. Recommend value is {}".format(
            int(100 // tc)), "red"))
    sim = model_settings.Cylinder(**kwargs)
    if init_f is not None:
        loaded_data = np.load('./data/init_frame.npz')
        init_f = resample_field(loaded_data['f'], shape=(nx, ny, loaded_data['f'].shape[2]))
        return sim.run_batch_generator(ts_end, ts_start, output_offset, output_stride, init_f=init_f)
    else:
        return sim.run_batch_generator(ts_end, ts_start, output_offset, output_stride)

def read_data(total_batch):
    res_data = {
        'timestep':[],
        'rho':[],
        'u':[]
    }
    for i in tqdm(range(total_batch)):
        loaded_data = np.load('./data/batched_ref_data_{}.npz'.format(i))
        res_data['timestep'].append(loaded_data['timestep'])
        res_data['rho'].append(loaded_data['rho'])
        res_data['u'].append(loaded_data['u'])
    print("concatenating ... ")
    res_data['timestep'] = np.concatenate(res_data['timestep'], axis=0)
    res_data['rho'] = np.concatenate(res_data['rho'], axis=0)
    res_data['u'] = np.concatenate(res_data['u'], axis=0)
    return res_data

def read_data_and_downsample(total_batch, factor=4):
    res_data = {
        'timestep':[],
        'rho':[],
        'u':[]
    }
    for i in tqdm(range(total_batch)):
        loaded_data = np.load('./data/batched_ref_data_{}.npz'.format(i))
        resized_shape = (loaded_data['u'].shape[0], round(loaded_data['u'].shape[1]/factor), round(loaded_data['u'].shape[2]/factor))
        res_data['timestep'].append(loaded_data['timestep'])
        res_data['rho'].append(resample_field(loaded_data['rho'], factor, shape=(resized_shape[0], resized_shape[1], resized_shape[2], loaded_data['rho'].shape[3])))
        res_data['u'].append(resample_field(loaded_data['u'], factor, shape=(resized_shape[0], resized_shape[1], resized_shape[2], loaded_data['u'].shape[3])))
    print("concatenating ... ")
    res_data['timestep'] = np.concatenate(res_data['timestep'], axis=0)
    res_data['rho'] = np.concatenate(res_data['rho'], axis=0)
    res_data['u'] = np.concatenate(res_data['u'], axis=0)
    return res_data

def generate_sim_dataset_and_save(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None, file_name="batched_ref_data"):
    def process_batched_data(batched_data, seq_num):
        batched_data['timestep'] = np.array(batched_data['timestep'])
        batched_data['u'] = np.stack(batched_data['u'], axis=0)
        batched_data['rho'] = np.stack(batched_data['rho'], axis=0)
        np.savez_compressed("./data/{}_{}.npz".format(file_name, seq_num), timestep=batched_data['timestep'], u=batched_data['u'], rho=batched_data['rho'])
        print("Seq {} saved".format(seq_num))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=24) as executor:
        seq = 0
        tasks = []
        for batched_data in batch_generator_generate_sim_dataset(diam, ts_start, ts_end, output_offset, output_stride, init_f):
            tasks.append(executor.submit(process_batched_data, batched_data, seq))
            seq += 1
            if len(tasks) == 10:
                print("Finishing tasks...")
                for task in tqdm(tasks):
                    task.result()
                tasks = []
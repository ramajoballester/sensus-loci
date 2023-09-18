import os
import unittest
import subprocess
import sensus


class TestAll(unittest.TestCase):

    def test_mmdet3d(self):
        # Test the correct installation of mmdet3d
        dest_path = sensus.__path__[0]
        config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car'
        command = f'mim download mmdet3d --config {config_file} --dest {dest_path}'
        subprocess.check_output(command, shell=True)
        # Remove the downloaded and generated file
        subprocess.check_output(f"rm {os.path.join(dest_path, '*pointpillars*.pth')}", shell=True)
        subprocess.check_output(f"rm {os.path.join(dest_path, config_file + '.py')}", shell=True)
        


if __name__ == '__main__':
    unittest.main()
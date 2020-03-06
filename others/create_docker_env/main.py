import subprocess
import shlex

cudnn_lib = {} #(cuda_ver, cudnn_ver)
cudnn_dev = {}

cudnn_lib[('9.0', '7.3')] = 'libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb'
cudnn_lib[('10.0', '7.3')] = 'libcudnn7_7.3.1.20-1+cuda10.0_amd64.deb'
cudnn_lib[('9.0', '7.4')] = 'libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb'
cudnn_lib[('10.0', '7.4')] = 'libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb'
cudnn_lib[('9.0', '7.5')] = 'libcudnn7_7.5.1.10-1+cuda9.0_amd64.deb'
cudnn_lib[('10.0', '7.5')] = 'libcudnn7_7.5.1.10-1+cuda10.0_amd64.deb'
cudnn_lib[('9.0', '7.6')] = 'libcudnn7_7.6.0.64-1+cuda9.0_amd64.deb'
cudnn_lib[('10.0', '7.6')] = 'libcudnn7_7.6.0.64-1+cuda10.0_amd64.deb'


cudnn_dev[('9.0', '7.3')] = 'libcudnn7-dev_7.3.1.20-1+cuda9.0_amd64.deb'
cudnn_dev[('10.0', '7.3')] = 'libcudnn7-dev_7.3.1.20-1+cuda10.0_amd64.deb'
cudnn_dev[('9.0', '7.4')] = 'libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb'
cudnn_dev[('10.0', '7.4')] = 'libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb'
cudnn_dev[('9.0', '7.5')] = 'libcudnn7-dev_7.5.1.10-1+cuda9.0_amd64.deb'
cudnn_dev[('10.0', '7.5')] = 'libcudnn7-dev_7.5.1.10-1+cuda10.0_amd64.deb'
cudnn_dev[('9.0', '7.6')] = 'libcudnn7-dev_7.6.0.64-1+cuda9.0_amd64.deb'
cudnn_dev[('10.0', '7.6')] = 'libcudnn7-dev_7.6.0.64-1+cuda10.0_amd64.deb'

docker_image = {}
docker_image['9.0'] = 'nvidia/cuda:9.0-devel-ubuntu16.04'
docker_image['10.0'] = 'nvidia/cuda:10.0-devel-ubuntu16.04'

def exec_cmd(cmd_str):
    print("Exec:", cmd_str)
    cmd = shlex.split(cmd_str)

    p = subprocess.Popen(cmd)
    p.communicate()


for cuda_ver in ['9.0', '10.0']:
    for cudnn_ver in ['7.3', '7.4', '7.5', '7.6']:

        print("Creating CUDA:", cuda_ver, "CUDNN:", cudnn_ver)

        docker_name = 'CUDA_' + cuda_ver + '_CUDNN_' + cudnn_ver
        image_name = 'cuda:cuda_' + cuda_ver + '_cudnn_' + cudnn_ver

        cmd_str = 'docker image rm ' + image_name
        exec_cmd(cmd_str)

        cmd_str = 'docker run -itd -v /home/user1/cudnn:/cudnn --name ' +  docker_name + ' ' + docker_image[cuda_ver] + ' bash'
        exec_cmd(cmd_str)

        cmd_str = 'docker start ' + docker_name
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' apt update'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' apt -yqq install python3-pip vim libgtk2.0-0 libgl1-mesa-glx openmpi-bin'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' mkdir /usr/local/mklml'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' tar -xzf /cudnn/mklml_lnx_2018.0.3.20180406.tgz -C /usr/local/mklml'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' apt -yqq install /cudnn/' + cudnn_lib[(cuda_ver, cudnn_ver)]
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' apt -yqq install /cudnn/' + cudnn_dev[(cuda_ver, cudnn_ver)]
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' deluser --quiet --force --remove-home --remove-all-files user2'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' bash -c "useradd -m -p $(openssl passwd -1 password) -s /bin/bash user2"'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' bash -c "echo \\"export LD_LIBRARY_PATH=/usr/local/mklml/mklml_lnx_2018.0.3.20180406/lib:/usr/local/anaconda3/lib:$LD_LIBRARY_PATH\\" >> /home/user2/.bashrc"'
        exec_cmd(cmd_str)

        cmd_str = 'docker exec ' + docker_name + ' bash -c "echo \\". /usr/local/anaconda3/etc/profile.d/conda.sh\\" >> /home/user2/.bashrc"'
        exec_cmd(cmd_str)

        cmd_str = 'docker stop ' + docker_name
        exec_cmd(cmd_str)

        cmd_str = 'docker container commit ' + docker_name + ' cuda:cuda_' + cuda_ver + '_cudnn_' + cudnn_ver
        exec_cmd(cmd_str)

        cmd_str = 'docker rm ' + docker_name
        exec_cmd(cmd_str)


        





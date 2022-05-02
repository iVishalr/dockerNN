#!/bin/bash
sudo apt update
sudo apt upgrade

echo "
Turning off swapping ..."
echo "Use the following command to turn it back on."
echo ""
echo "  sudo swapon -a"

sudo swapoff -a

read -e -p "
Do you want to install docker-engine and docker-cli? [Y/n] " YN

[[ $YN == "y" || $YN == "Y" || $YN == "" ]] && \
sudo apt install apt-transport-https ca-certificates curl software-properties-common && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && \
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" && \
sudo apt-get install docker-ce docker-ce-cli containerd.io -y && \
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
} 
EOF&&\
sudo systemctl enable docker && sudo systemctl daemon-reload && sudo systemctl restart docker

echo "
Please ensure you do not have minikube or any kubernetes emulator installed. Kubernetes Cluster setup may not be successful if present." 

read -e -p "
Do you want to install kubelet kubeadm kubectl? [Y/n] " YN
[[ $YN == "y" || $YN == "Y" || $YN == "" ]] && \
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg && \
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list && \
sudo apt-get update -y && \
sudo apt-get install -y kubelet kubeadm kubectl && \
sudo apt-mark hold kubelet kubeadm kubectl

read -e -p "
Please enter the IP Address of your system : " IPADDR

NODENAME=$(hostname -s)
echo "Initializing Master Node Control Plane ..."
sudo kubeadm init --apiserver-advertise-address=$IPADDR --apiserver-cert-extra-sans=$IPADDR --pod-network-cidr=192.168.0.0/16 --node-name $NODENAME --ignore-preflight-errors Swap

sudo rmdir -p $HOME/.kube
sudo mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

read -e -p "
Do you want to schedule apps on master node? [Y/n] " YN

[[ $YN == "y" || $YN == "Y" || $YN == "" ]] && \
kubectl taint nodes --all node-role.kubernetes.io/master-

kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# read -e -p "
# Do you want to test kubenetes cluster setup? [Y/n] " YN

# [[ $YN == "y" || $YN == "Y" || $YN == "" ]] && \
# cat <<EOF | kubectl apply -f -
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   name: nginx-deployment
# spec:
#   selector:
#     matchLabels:
#       app: nginx
#   replicas: 2 
#   template:
#     metadata:
#       labels:
#         app: nginx
#     spec:
#       containers:
#       - name: nginx
#         image: nginx:latest
#         ports:
#         - containerPort: 80      
# EOF && \
# cat <<EOF | kubectl apply -f -
# apiVersion: v1
# kind: Service
# metadata:
#   name: nginx-service
# spec:
#   selector: 
#     app: nginx
#   type: NodePort  
#   ports:
#     - port: 80
#       targetPort: 80
#       nodePort: 32000
# EOF && \
# echo "Nginx Service Running! Please access it by typing http://$IPADDR:32000"
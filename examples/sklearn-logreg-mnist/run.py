import subprocess
import time
import argparse
import random

def launch_scripts(num_rounds, num_clients, random_clients, num_classes, random_distribution, offset):
    # Launch server.py
    server = subprocess.Popen(["python", "server.py", "--num_rounds", str(num_rounds)])

    # Wait for the server to start
    time.sleep(3)

    # Launch num_clients instances of client.py
    for i in range(num_clients):
        # Determine the number of classes
        if random_class_num:
            num_classes = random.randint(1, 10)

        if random_distribution:
            # n classes are randomly chosen from the 10 classes
            class_subset = random.sample(range(10), num_classes)
        else:
            # n classes are chosen from the 10 classes with an offset between the classes for each client
            class_subset = [j % 10 for j in range(i * offset, i * offset + num_classes)]
        print(f"Launching client with classes {class_subset}")
        client = subprocess.Popen(["python", "client.py", "--classes"] + [str(c) for c in class_subset])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", "-r", type=int, required=True)
    parser.add_argument("--num_clients", "-c", type=int)
    class_args = parser.add_mutually_exclusive_group(required=True)
    class_args.add_argument("--num_classes", "-n", type=int, default=1)
    class_args.add_argument("--random_class_num", action='store_true', help="Generate a random number of classes")
    dist_args = parser.add_mutually_exclusive_group(required=True)
    dist_args.add_argument("--random_distribution", action='store_true')
    dist_args.add_argument("--offset", "-o", type=int)
    args = parser.parse_args()

    launch_scripts(args.num_rounds, args.num_clients, args.random_class_num, args.num_classes, args.random_distribution, args.offset)
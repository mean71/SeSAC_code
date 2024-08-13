import random
import math

from ADT.queue import Queue
from ADT.stack import Stack  

class Customer:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.service_end_time = None

    def wait_time(self):
        return self.service_start_time - self.arrival_time if self.service_start_time else 0

    def service_time(self):
        return self.service_end_time - self.service_start_time if self.service_end_time else 0

class Bank:
    def __init__(self, num_tellers, service_mean, service_stddev):
        self.num_tellers = num_tellers
        self.service_mean = service_mean
        self.service_stddev = service_stddev
        self.teller_available_times = [0] * num_tellers  # Time when each teller will be free

    def process_customer(self, customer, current_time):
        pass 

    def run(self, customers):
        pass 

def generate_customers(arrival_rate, simulation_time):
    customers = []
    current_time = 0
    while current_time < simulation_time:
        # Generate inter-arrival time based on Poisson distribution
        inter_arrival_time = random.expovariate(arrival_rate)
        current_time += inter_arrival_time
        if current_time < simulation_time:
            customers.append(Customer(current_time))
    return customers

def run_simulation(num_tellers, arrival_rate, service_mean, service_stddev, simulation_time):
    # Generate customers
    customers = generate_customers(arrival_rate, simulation_time)

    # Initialize bank and process customers
    bank = Bank(num_tellers, service_mean, service_stddev)
    bank.run(customers)


if __name__ == '__main__':
    # Parameters for the simulation
    num_tellers = 3                # Number of tellers
    arrival_rate = 2.0             # Average number of customers arriving per minute (Poisson distribution)
    service_mean = 4.0             # Mean service time in minutes (Normal distribution)
    service_stddev = 1.0           # Standard deviation of service time (Normal distribution)
    simulation_time = 20           # Total simulation time in minutes

    # Run the simulation
    run_simulation(num_tellers, arrival_rate, service_mean, service_stddev, simulation_time)

import multiprocessing as mp
import pickle
from manager.main_manager import Manager
import Interface.app as intf_app
import manager.Message_types as msg_types
from copy import deepcopy
import random
import os
import time
import datetime

SERVER_ID = 0
MANAGER_TIMEOUT_SECS = 30*60
SERVER_TIMEOUT_SECS = 30*60
PLANS_PER_ROUND = 16
final_pickle_file_name = "super_manager_pickle.p"
#todo NOTE to kill tcp processes enter in terminal "fuser -k 5000/tcp"
#======================================================================================
def parallel_dummy_client(client_id, super_pipe):
    """

    :param input_manager:
    :param comm_pipe:
    :return:
    """
    #run the flask app
    super_pipe.send([msg_types.from_client.NEW_CLIENT_REQUEST, client_id])
    super_pipe.send([msg_types.from_client.REQUEST_NEW_PLANS, client_id])
    time.sleep(20)
    super_pipe.send([msg_types.from_client.REQUEST_NEW_PLANS, client_id])
    time.sleep(20)
    super_pipe.send([msg_types.from_client.REQUEST_NEW_PLANS, client_id])
    time.sleep(20)
    super_pipe.send([msg_types.from_client.ANNOTATED_PLANS, client_id])

    # SEE IF THE CORE QUERY HAPPENS, AND IF THE RBUS IS RUN
    # SEND ONE MORE REQUEST FOR PLANS AND SEE IF YOU GET THE RBUS PLANS


    while True:
        time.sleep(60)



#===================================================================

def parallel_server_starter(starting_client_id, comm_pipe,multiprocess_msg_dict):
    """

    :param input_manager:
    :param comm_pipe:
    :return:
    """
    #run the flask app
    intf_app.next_user_id = starting_client_id
    intf_app.super_manager_comm_pipe = comm_pipe
    intf_app.multiprocess_msg_list_dict = multiprocess_msg_dict
    print(" parallel server starter ", os.getpid())
    intf_app.app.run(debug=True)

#===================================================================

def parallel_server_function_handler( comm_pipe,multiprocess_msg_dict):
    """

    :param input_manager:
    :param comm_pipe:
    :return:
    """
    #run the flask app
    while True:
        ready_connections_objects_list = mp.connection.wait([comm_pipe],timeout=SERVER_TIMEOUT_SECS)
        if comm_pipe in ready_connections_objects_list:
            print("server function handler received a msg")
            super_manager_msg = comm_pipe.recv()
            if super_manager_msg[0] == msg_types.to_client.ACK_CLIENT_REQ:
                #todo NOTE do not just append, the changes will not propagate through the multiprocess manager
                # need to reassign, not just modify mutable objects, only then will the changes propagate
                multiprocess_msg_dict[super_manager_msg[1]] += [ [super_manager_msg[0]] ] #list of messages, each message is a list as well
            elif super_manager_msg[0] == msg_types.to_client.PLANS_FOR_ANNOTATION:
                print("REACHED sending msg_types.to_client.PLANS_FOR_ANNOTATION",time.time())
                print([super_manager_msg[0] , super_manager_msg[2]])
                #Todo NOTE: Each message is a list. these 'list' messages are in a larger list
                # the larger list is stored at multiprocess_msg_dict[super_manager_msg[1]]. This needs to be reassigned for the changed to propagate
                # through the multiprocess manager that handles this multiprocess dictionary. Hence we do "+=" rather than append a single new message
                multiprocess_msg_dict[super_manager_msg[1]] += [ [super_manager_msg[0] , super_manager_msg[2]] ]
            elif super_manager_msg[0] == msg_types.to_client.COMPLETED:
                multiprocess_msg_dict[super_manager_msg[1]] +=  [ [msg_types.to_client.COMPLETED] ]#empty plans list when completed


#===================================================================

def parallel_manager_function_handler(manager, comm_pipe,client_id):
    """
    :summary : handles the subprocess computation
    :param manager:
    :param comm_pipe:
    :return:
    """
    #while true the function runs. It waits to receive a message from the super manager. When we get exit message,
    # the pickled manager is sent back and we exit
    manager.prep_next_round_plans_for_intf()
    while True:
        ready_connections_objects_list = mp.connection.wait([comm_pipe],timeout=MANAGER_TIMEOUT_SECS)
        if comm_pipe not in ready_connections_objects_list:
            print(" TIMEOUT IN MANAGER SENT COMPLETED for client ",client_id)
            comm_pipe.send([msg_types.from_manager.PLANS_FOR_ANNOTATION, client_id,[]])  # EMPTY LIST IS END OF PROCESS, will be sent to the gui
            comm_pipe.send([msg_types.from_manager.COMPLETED, client_id, manager])
            break;  # exit the while true loop and end the process
        else:
            super_manager_msg = comm_pipe.recv()
            if super_manager_msg[0] == msg_types.to_manager.REQUEST_NEW_PLANS:
                print("got request for new plans",time.time())
                if manager.completed: #todo NOTE the +1 is for the test set of plans.
                    #TEST plan set is extracted ahead of time and then kept aside from the start. It is just sent at the end
                    print(" SENT COMPLETED FROM MANAGER")
                    comm_pipe.send( [msg_types.from_manager.PLANS_FOR_ANNOTATION, client_id, []]) #EMPTY LIST IS END OF PROCESS, will be sent to the gui

                    # todo print the results into a text file, NEEDS MORE CODE
                    #pickle the ENTIRE manager into a file, that way we can calmly get other stats as needed
                    with open("RESULTS_CLIENT_ID_"+str(client_id)+"_"+str(datetime.datetime.now())+".p","wb") as dest:
                        pickle.dump(manager,dest)
                    comm_pipe.send([msg_types.from_manager.COMPLETED,client_id,manager])
                    break;  #exit the while true loop and end the process
                #end if
                send_msg = [msg_types.from_manager.PLANS_FOR_ANNOTATION, client_id, manager.get_next_round_plans_for_intf()]
                comm_pipe.send(send_msg)
                print("SENT plans from manager",time.time())
                if manager.check_next_round_RBUS(): #query for cores before using them to prep plans
                    print("Sending query cores availability")
                    comm_pipe.send([msg_types.from_manager.QUERY_CORES_AVAILABILITY,client_id])
                else: #it is a dbs round, just use one core to do so
                    print("prepping next round",time.time())
                    manager.prep_next_round_plans_for_intf()
            elif super_manager_msg[0] == msg_types.to_manager.CORES_AVAILABILITY:
                print("received cores availability and prepping next rbus round of plan",time.time())
                num_cores = max(1,super_manager_msg[1])#we would have requested cores after sending DBS, and so we would
                manager.set_num_cores_RBUS(num_cores)# have processed the CORES AVAILABLE msg before PROCESS_ANNOTATED_PLANS
                manager.prep_next_round_plans_for_intf()
            elif super_manager_msg[0] == msg_types.to_manager.PROCESS_ANNOTATED_PLANS:
                print("Received annot plans for processing")
                print(super_manager_msg)

                if manager.curr_round > manager.num_rounds + 1:
                    manager.store_annot_test_set(super_manager_msg[1])
                    manager.evaluate()
                else:
                    manager.reformat_features_and_update_indices(super_manager_msg[1])
                    # if not manager.check_next_round_RBUS() or manager.curr_round == 2 :#or manager.curr_round > manager.num_rounds:# 2 means dbs after exploration, otherwise do not learn the model after the dbs round, wait for the rbus round as well
                    #     manager.relearn_model(learn_LSfit=True, num_chains=2)  # MLE model learned for evaluation

            elif super_manager_msg[0] == msg_types.to_manager.FORCE_END:
                break; #DO NOT SAVE THIS MANAGER. ENDED BADLY

                # if manager.check_next_round_RBUS():
                #     manager.prep_next_round_plans_for_intf() #for the just send DBS round
            # elif super_manager_msg[0] == msg_types.to_manager.DELETE_sim_human_annotate:
            #     print("running PSEUDO ANNOTATIONS, should NOT BE THERE ERROR")
            #     print(super_manager_msg[1])
            #     annot_plans = manager.get_feedback(super_manager_msg[1])
            #     manager.update_indices(annot_plans)
            #     manager.relearn_model(learn_LSfit=True, num_chains=2)  # MLE model learned for evaluation
                #THIS AUTOMATICALLY calls evaluate, and stores the results

#===================================================================
class SuperManager:

    def __init__(self,init_pickle_file = None, num_experiments = 100):
        """
        :summary:
        """

        self.default_manager = None
        self.starting_client_id = SERVER_ID
        self.num_experiments = 100
        self.completed_managers = []

        # todo NOTE better to always create a new manager
        # # next_client_id is managed by the app script 0 is given to the default manager which is deepcopied for every new client
        if init_pickle_file == None:
            init_pickle_file = "default_manager_pickle_file.p"
        try:
            with open(init_pickle_file,"rb") as src:
                self.starting_client_id = pickle.load(src)
                self.default_manager = pickle.load(src)
        except:
            print("CREATING NEW DEFAULT MANAGER")
            self.starting_client_id = 1
            self.default_manager = Manager(num_rounds = 3,num_dataset_plans = 10000,with_simulated_human = False,
                 max_feature_size = 5,FEATURE_FREQ_CUTOFF = 0.05,pickle_file_for_plans_pool = "default_plans_pool.p",
                                           test_set_size=20,plans_per_round=PLANS_PER_ROUND)
            with open(init_pickle_file,"wb") as dest:
                self.starting_client_id = pickle.dump(self.starting_client_id,dest)
                self.default_manager = pickle.dump(self.default_manager,dest)

        # if self.default_manager == None:
        #     self.starting_client_id = SERVER_ID
        #     self.default_manager = Manager(num_rounds = 2, num_dataset_plans = 10000, with_simulated_human = False,
        #          max_feature_size = 5, FEATURE_FREQ_CUTOFF = 0.05, pickle_file_for_plans_pool = "default_plans_pool.p",
        #                                    test_set_size=20, plans_per_round=10)
        #     # no need to pickle ? always create a new one
        #     with open(init_pickle_file,"wb") as dest:
        #         self.starting_client_id = pickle.dump(self.starting_client_id,dest)
        #         self.default_manager = pickle.dump(self.default_manager,dest)

        #end if
        self.data_multiprocess_manager = mp.Manager()
        self.multiprocess_msg_dict = self.data_multiprocess_manager.dict()
        self.multiprocess_msg_dict[0] = ["TESTING"]
        self.manager_dict = {}
        self.process_dict = {}
        self.server_pipe =  None
        self.connection_object_list = []

        # todo uncomment this
        self.create_server()
        # self.create_dummy_client(1)

        self.message_wait_loop()
        self.__del__()

    # ===================================================================

    def __del__(self):
        """
        :summary: Pickle all the managers and the next clientID. CLIENT id IS FIRST. Tells us how many managers are there as well
        :return:
        """
        for process_key in self.process_dict.keys():
            # send the exit message and get all managers to pickle
            try:
                self.process_dict[process_key][0].join()
            except:
                pass
        # end for loop
    # end destructor function

    # ===================================================================
    def create_dummy_client(self, client_id):
        """

        :return:
        """

        #create a dummy client along with it's manager
        serverPipe, destPipe = mp.Pipe()
        self.server_pipe = serverPipe
        self.connection_object_list.append(serverPipe)
        new_process = mp.Process(target=parallel_dummy_client,args=( client_id,destPipe))

        new_process.daemon = True #so if the parent dies, it kills the children too.
        new_process.start()
        self.process_dict[-1] = [new_process, None]

        sourcePipe, destPipe = mp.Pipe()
        new_manager = deepcopy(self.default_manager)
        self.manager_dict[client_id] = sourcePipe
        self.connection_object_list.append(sourcePipe)
        new_process = mp.Process(target=parallel_manager_function_handler,args=(new_manager, destPipe, client_id))
        new_process.start()
        self.process_dict[client_id] = [new_process, destPipe]

    # ===================================================================
    def create_manager(self,client_id):
        """

        :return:
        """
        sourcePipe, destPipe = mp.Pipe()
        new_manager = deepcopy(self.default_manager)
        self.manager_dict[client_id] = sourcePipe
        self.connection_object_list.append(sourcePipe)
        print("MANAGER ", new_manager.plans_per_round)
        new_process = mp.Process(target=parallel_manager_function_handler, args=(new_manager,destPipe,client_id))
        new_process.start()
        self.process_dict[client_id] = [new_process, destPipe]

    # ===================================================================
    def create_server(self):
        """
        :return:
        """
        print("server created")
        sourcePipe, destPipe = mp.Pipe()
        self.server_pipe = sourcePipe
        self.connection_object_list.append(sourcePipe)
        #start the server function handler first so we are ready to process messages, before starting the app
        new_process = mp.Process(target=parallel_server_function_handler, args=(destPipe,self.multiprocess_msg_dict))
        new_process.start()
        new_process = mp.Process(target=parallel_server_starter, args=(self.starting_client_id, destPipe,self.multiprocess_msg_dict))
        new_process.start()
        self.process_dict[SERVER_ID] = [new_process, destPipe]

    # ===================================================================
    def message_wait_loop(self):
        """
        :summary: The main function that waits on messages
        :return:
        """
        while True:
            ready_connections_objects_list = mp.connection.wait(self.connection_object_list)
            #PRIORITY is given to the server
            if self.server_pipe in ready_connections_objects_list:
                print("got a message from server ",time.time())
                recv_msg = self.server_pipe.recv()
                self.process_server_message(recv_msg)
            else:  # it is a message from a manager
                #select a random connection object to process
                random_pipe = random.choice(ready_connections_objects_list)
                self.process_manager_message(random_pipe.recv())
            if len(self.completed_managers) == self.num_experiments:
                break #out of the while loop, we are done
        #end while true loop



    # ===================================================================
    def process_server_message(self,recv_msg):
        """

        :param self:
        :return:
        """

        if recv_msg[0] == msg_types.from_client.NEW_CLIENT_REQUEST:
            self.create_manager(recv_msg[1])
            print("manager created")
            self.server_pipe.send([msg_types.to_client.ACK_CLIENT_REQ, recv_msg[1], "CLIENT request acknowledged"])
        elif recv_msg[0] == msg_types.from_client.REQUEST_NEW_PLANS:
            client_id = recv_msg[1]
            dest_pipe = self.manager_dict[client_id]
            dest_pipe.send([msg_types.to_manager.REQUEST_NEW_PLANS])
        elif recv_msg[0] == msg_types.from_client.ANNOTATED_PLANS:
            client_id = recv_msg[1]
            dest_pipe = self.manager_dict[client_id]
            dest_pipe.send([msg_types.to_manager.PROCESS_ANNOTATED_PLANS,recv_msg[2]])
        elif recv_msg[0] == msg_types.from_client.FORCE_END:
            client_id = recv_msg[1]
            dest_pipe = self.manager_dict[client_id]
            dest_pipe.send([msg_types.to_manager.FORCE_END, recv_msg[2]])
            try:
                self.process_dict[client_id][0].terminate()
                self.process_dict[client_id][0].join()
                del self.process_dict[client_id]
            except KeyError:
                pass



    #===================================================================

    def process_manager_message(self,recv_msg):
        """
        :return:
        """
        if recv_msg[0] == msg_types.from_manager.PLANS_FOR_ANNOTATION:
            print("SENT plans to client for annotation",time.time())
            dest_client_id = recv_msg[1]
            payload = recv_msg[2]
            self.server_pipe.send([msg_types.to_client.PLANS_FOR_ANNOTATION] + recv_msg[1:])
            # TODO DELETE this temp score
            # self.manager_dict[recv_msg[1]].send([msg_types.to_manager.DELETE_sim_human_annotate, payload])
        elif recv_msg[0] == msg_types.from_manager.COMPLETED:
            dest_client_id = recv_msg[1]
            payload = recv_msg[2]
            self.server_pipe.send([msg_types.to_client.COMPLETED,dest_client_id])
            completed_manager = payload
            if completed_manager.completed: #then the manager did all rounds including test set
                self.completed_managers.append(completed_manager) #in the message wait loop, it will exit
                with open("TEMP_completed_managers_storage.p","wb") as dest:
                    pickle.dump(self.completed_managers,dest)
                #end with
            #end if the manager successfully completed
            self.process_dict[dest_client_id][0].join() #end that process
        elif recv_msg[0] == msg_types.from_manager.QUERY_CORES_AVAILABILITY:
            self.manager_dict[recv_msg[1]].send([msg_types.to_manager.CORES_AVAILABILITY,mp.cpu_count()-1])
        #end if

#============================================
if __name__ == "__main__":
    superman = SuperManager()
    with open(final_pickle_file_name,"wb") as dest:
        pickle.dump(superman,dest)

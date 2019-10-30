
from enum import Enum



class to_client(Enum):
    ACK_CLIENT_REQ = 1 #the app file assigns the client id
    PLANS_FOR_ANNOTATION = 2
    COMPLETED = 3 #this message is NOT sent to the app, instead we sent an empty plans for annotation.


class from_client(Enum):
    NEW_CLIENT_REQUEST = 1 #the app file assigns the client ID
    REQUEST_NEW_PLANS = 2
    ANNOTATED_PLANS = 3
    FORCE_END = 4 # if we got an empty set of plans and didn't get a complete


class to_manager(Enum):
    NEW_CONNECTION = 1
    REQUEST_NEW_PLANS = 2
    PROCESS_ANNOTATED_PLANS = 3
    CORES_AVAILABILITY = 4
    FORCE_END = 5
    DELETE_sim_human_annotate = 6


class from_manager(Enum):
    PLANS_FOR_ANNOTATION = 1
    QUERY_CORES_AVAILABILITY = 2
    COMPLETED = 3 #WHEN the super manager asks for the next set of plans, and the manager has finished, it will return
                    #that it completed. At which point the super manager will send a close connection message
                #note the super manager can extend the number of rounds if the client is willing to do so.
                # it would simply increase the number of rounds, and ask for plans to the manager again.


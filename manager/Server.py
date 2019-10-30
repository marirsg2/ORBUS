import select
import socket
import sys
"""
:NOTES: This server class is to enable IP communication with the client.
"""

class Server:
    MAX_CONNECTIONS = 32

    def __init__(self, sock =None, server_address = ('localhost', 10000)):
        """
        :summary:
        """
        if sock is None:
            self.server_sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.server_sock = sock
        self.server_sock.setblocking(0) #NON BLOCKING
        self.server_sock.bind(server_address)
        self.server_sock.listen(Server.MAX_CONNECTIONS)
        self.input_sockets = [self.server_sock]
        self.output_sockets = [] #client sockets
        self.to_send_msg_list = []
        self.received_msg_list = []

        pass

    # ===================================================================

    def __del__(self):
        """

        :return:
        """
        pass
    # ===================================================================

    def process_ip_socket(self):
        """

        :return:
        """
        input_ready,output_ready,exceptions_list = \
            select.select(self.input_sockets,self.output_sockets,self.input_sockets,timeout=1)
        #check for any received messages,
        self.receive_messages()
        #send any messages pending
        self.send_messages()
        pass

    # ===================================================================

    def connect(self, host, port):
        """
        WILL not be used
        :param host:
        :param port:
        :return:
        """
        self.server_sock.connect((host, port))
    # ===================================================================


    def receive_messages(self):
        """

        :return:
        """
        pass

    # ===================================================================

    def send_messages(self):
        """

        :return:
        """
        pass
    # ===================================================================
    def process_super_manager_message(self,super_msg):
        """

        :return:
        """
        #find which client to ADD to the output sockets
        pass
    # ===================================================================

import __nodos_internal__ as nos_py
from typing import Any
from abc import ABC, abstractmethod

NODOS_PYTHON_API_VERSION_MAJOR = 0
NODOS_PYTHON_API_VERSION_MINOR = 1


class result:
    """
    Nodos result class
    """
    SUCCESS = 0
    FAILURE = 1


class uuid:
    """
    Nodos uuid class
    """

    def __init__(self):
        pass


def log_info(message: str):
    """
    Log an information message

    :param message: The message to log
    """
    nos_py.log_info(message)


def log_warning(message: str):
    """
    Log a warning message

    :param message: The message to log
    """
    nos_py.log_warning(message)


def log_error(message: str):
    """
    Log an error message

    :param message: The message to log
    """
    nos_py.log_error(message)


def set_pin_value(pin_id: Any, value: Any):
    """
    Set the value of a pin

    :param pin_id: The unique identifier of the pin
    :param value: The value to set
    """
    nos_py.set_pin_value(pin_id, value)


class NodeExecuteArgs(ABC):
    """
    Nodos NodeExecuteArgs class
    """

    @property
    def node_class_name(self) -> str:
        """
        Get the name of the class of the node

        :return: The name of the class of the node
        """
        pass

    @property
    def node_name(self) -> str:
        """
        Get the name of the node

        :return: The name of the node
        """
        pass

    @abstractmethod
    def get_pin_value(self, pin_name: str) -> Any:
        """
        Access the memory of the pin specified by 'pin_name'

        :param pin_name: The name of the pin
        :return: The value of the pin
        """
        pass

    @abstractmethod
    def get_pin_id(self, pin_name: str) -> int:
        """
        Get the unique identifier of the pin specified by 'pin_name'

        :param pin_name: The name of the pin
        :return: The unique identifier of the pin
        """
        pass


class PinConnectedArgs(ABC):
    """
    Nodos PinConnectedArgs class
    """

    @property
    def pin_name(self) -> str:
        """
        :return: The name of the pin connected to your node

        """
        pass


class PinDisconnectedArgs(ABC):
    """
    Nodos PinDisconnectedArgs class
    """

    @property
    def pin_name(self) -> str:
        """
        :return: The name of the pin disconnected from your node
        """
        return self.pin_name


class PinValueChangedArgs(ABC):
    """
    Nodos PinValueChangedArgs class
    """

    @property
    def pin_name(self) -> str:
        """
        :return: The name of the pin whose value has changed
        """
        return self.pin_name

    @property
    def pin_value(self) -> Any:
        """
        :return: The value of the pin
        """
        return self.pin_value


class ContextMenuRequestInstigator:
    @property
    def editor_id(self) -> int:
        """
        :return: The unique identifier of the editor that requested the context menu
        """
        return self.editor_id

    @property
    def request_id(self) -> int:
        """
        :return: The unique identifier of the request
        """
        return self.request_id


class ContextMenuRequest:
    @property
    def item_id(self) -> uuid:
        """
        :return: The unique identifier of the item that requested the context menu
        """
        return self.item_id

    @property
    def position(self) -> Any:
        """
        :return: The position of the context menu. 
                 Should be echoed back when sending a context menu update response.
        """
        return self.position

    @property
    def instigator(self) -> ContextMenuRequestInstigator:
        """
        :return: The instigator of the context menu request. 
                 Should be echoed back when sending a context menu update response.
        """
        return self.instigator


class Node(ABC):
    """
    Nodos Node class
    """

    def on_node_created(self, args: Any):
        pass

    def on_pin_value_changed(self, args: NodeExecuteArgs):
        pass

    def on_pin_connected(self, args: PinConnectedArgs):
        pass

    def on_pin_disconnected(self, args: PinDisconnectedArgs):
        pass

    def execute_node(self, args: NodeExecuteArgs):
        pass

    def on_node_menu_requested(self, request: ContextMenuRequest):
        pass

    def on_pin_menu_requested(self, request: ContextMenuRequest):
        pass

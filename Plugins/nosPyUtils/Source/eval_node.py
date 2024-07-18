import nodos as nos
import struct
from typing import Any

class Eval(nos.Node):
    def __init__(self):
        pass
    def on_pin_value_changed(self, args: nos.PinValueChangedArgs):
        pass

    def execute_node(self, args: nos.NodeExecuteArgs):
        str_mem: memoryview  = args.get_pin_value("Expression")
        nos.log_info(f"Input: Size {str_mem.nbytes} bytes")
        byte_array = bytearray(str_mem.tobytes())
        # Remve trailing zeros
        byte_array = byte_array.rstrip(b'\x00')
        str_mem = memoryview(byte_array)
        str_val = str_mem.tobytes().decode('utf-8')
        result = eval(str_val)
        nos.log_info(f"Evaluated expression: {str_val} = {result}")
        return nos.result.SUCCESS
    def on_node_menu_requested(self, request: nos.ContextMenuRequest):
        nos.log_info(f"Node menu requested for: {request.item_id}")
import sys

sys.path.append("./iris/experimental")

import my_module as anvil

print("Get isntance")

instance = anvil.AnvilLib.get_instance()
print("initialize")
instance.init()

print("Connect 0 to 1")

instance.connect(0, 1, 1)

queue = instance.get_sdma_queue(0, 1, 0)

# handle = queue.device_handle()

handle = anvil.get_handle_as_tensor(queue)

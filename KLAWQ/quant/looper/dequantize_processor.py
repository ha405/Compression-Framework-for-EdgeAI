from typing import Dict

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..utils.logger import setup_logger

log = setup_logger()

class DequantizeProcessor(LoopProcessor):
    def __init__(self, quantized_modules: Dict[str, TorchQuantLinear]):
        super().__init__(tokenizer=None, qcfg=None, calibration_dataset=None, calibration_dataset_concat_size=None,
                         prepare_dataset_func=None, batch_size=1,
                         logger_board="", require_fwd=False)

        self.quantized_modules = quantized_modules

    def set_calibration_dataset(self, calibration_dataset):
        self.calibration_dataset = None
        self.num_batches = 0

    # de-quantize weights
    def process(self, module: NamedModule, auto_gc: bool = True):
        device = module.weight.device
        w = module.weight.data

        # TODO fix num_itr param..need to calculate this before dequant
        m = self.quantized_modules.pop(module.full_name)
        m.optimize()
        log.info(f"Dequantize: `{m.name}`")
        wq = m.dequantize_weight().T.to(device=device)

        module.state.update({
            "w": w,
            "wq": wq,
        })

    def submodule_finalize(self, module: NamedModule):
        module.state.pop("w", None)  # no need for these weights now
        module.state.pop("wq", None) # no need for these weights now

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        return False

    def name(self) -> str:
        return "de-quantize"

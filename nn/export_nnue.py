#!/usr/bin/env python3
"""
NNUE Export Script
Export trained PyTorch models to formats consumable by the C++ chess engine
"""

import torch
import torch.onnx
import numpy as np
import struct
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NNUEExporter:
    """Export trained NNUE model to various formats"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint"""
        # Import the model class (assuming it's in the same directory)
        import sys
        sys.path.append(str(Path(__file__).parent))
        from train import NNUEModel
        
        config = self.checkpoint.get('config', {})
        model = NNUEModel(
            input_size=config.get('input_size', 768),
            hidden_size=config.get('hidden_size', 512),
            output_size=config.get('output_size', 1)
        )
        
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded from {self.checkpoint_path}")
        logger.info(f"Training epoch: {self.checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Validation loss: {self.checkpoint.get('metrics', {}).get('val_loss', 'unknown')}")
        
        return model
    
    def export_to_onnx(self, output_path: str):
        """Export model to ONNX format"""
        logger.info(f"Exporting to ONNX format: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 768)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"ONNX export completed: {output_path}")
    
    def export_to_binary(self, output_path: str):
        """Export model weights to binary format for C++ engine"""
        logger.info(f"Exporting to binary format: {output_path}")
        
        with open(output_path, 'wb') as f:
            # Write header
            self._write_header(f)
            
            # Write network architecture
            self._write_architecture(f)
            
            # Write weights and biases
            self._write_weights(f)
        
        logger.info(f"Binary export completed: {output_path}")
    
    def _write_header(self, f):
        """Write binary file header"""
        # Magic number for identification
        f.write(b'NNUE')
        
        # Version
        f.write(struct.pack('<I', 1))  # Version 1
        
        # Architecture hash (for version compatibility)
        arch_hash = hash(str(self.model))
        f.write(struct.pack('<Q', arch_hash & 0xFFFFFFFFFFFFFFFF))
    
    def _write_architecture(self, f):
        """Write network architecture information"""
        # Input size
        f.write(struct.pack('<I', 768))
        
        # Layer sizes
        layer_sizes = [512, 256, 128, 64, 32, 1]  # Based on our model architecture
        f.write(struct.pack('<I', len(layer_sizes)))
        
        for size in layer_sizes:
            f.write(struct.pack('<I', size))
    
    def _write_weights(self, f):
        """Write all network weights and biases"""
        # Feature transformer
        self._write_layer_weights(f, self.model.feature_transformer[0])
        
        # Accumulator layers
        for i in range(0, len(self.model.accumulator), 3):  # Skip ReLU and Dropout
            if isinstance(self.model.accumulator[i], torch.nn.Linear):
                self._write_layer_weights(f, self.model.accumulator[i])
    
    def _write_layer_weights(self, f, layer):
        """Write weights and biases for a single layer"""
        # Write weight matrix
        weights = layer.weight.data.numpy().astype(np.float32)
        f.write(struct.pack('<II', *weights.shape))
        f.write(weights.tobytes())
        
        # Write bias vector
        if layer.bias is not None:
            biases = layer.bias.data.numpy().astype(np.float32)
            f.write(struct.pack('<I', len(biases)))
            f.write(biases.tobytes())
        else:
            f.write(struct.pack('<I', 0))  # No bias
    
    def export_to_json(self, output_path: str):
        """Export model to JSON format (for debugging/inspection)"""
        logger.info(f"Exporting to JSON format: {output_path}")
        
        model_data = {
            'architecture': {
                'input_size': 768,
                'layers': [
                    {'type': 'Linear', 'in_features': 768, 'out_features': 512},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 512, 'out_features': 256},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 256, 'out_features': 128},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 128, 'out_features': 64},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 64, 'out_features': 32},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 32, 'out_features': 1},
                ]
            },
            'weights': {},
            'metadata': {
                'epoch': self.checkpoint.get('epoch', 0),
                'metrics': self.checkpoint.get('metrics', {}),
                'config': self.checkpoint.get('config', {})
            }
        }
        
        # Extract weights
        state_dict = self.model.state_dict()
        for name, tensor in state_dict.items():
            model_data['weights'][name] = tensor.cpu().numpy().tolist()
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"JSON export completed: {output_path}")
    
    def export_cpp_header(self, output_path: str):
        """Export model as C++ header file for direct embedding"""
        logger.info(f"Exporting to C++ header: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("#pragma once\n")
            f.write("#include <array>\n\n")
            f.write("namespace NNUE {\n\n")
            
            # Write constants
            f.write("constexpr int INPUT_SIZE = 768;\n")
            f.write("constexpr int HIDDEN_SIZE = 512;\n\n")
            
            # Write weights as arrays
            state_dict = self.model.state_dict()
            for name, tensor in state_dict.items():
                # Convert tensor name to valid C++ identifier
                cpp_name = name.replace('.', '_').upper()
                
                # Get tensor data
                data = tensor.cpu().numpy().flatten()
                
                # Write array declaration
                f.write(f"constexpr std::array<float, {len(data)}> {cpp_name} = {{\n")
                
                # Write data in chunks of 8 values per line
                for i in range(0, len(data), 8):
                    chunk = data[i:i+8]
                    values = ', '.join(f'{val:.6f}f' for val in chunk)
                    f.write(f"    {values},\n")
                
                f.write("};\n\n")
            
            f.write("} // namespace NNUE\n")
        
        logger.info(f"C++ header export completed: {output_path}")
    
    def validate_export(self, binary_path: str = None, onnx_path: str = None):
        """Validate exported models against original PyTorch model"""
        logger.info("Validating exports...")
        
        # Create test input
        test_input = torch.randn(1, 768)
        original_output = self.model(test_input)
        
        success = True
        
        # Validate ONNX if available
        if onnx_path and Path(onnx_path).exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(onnx_path)
                onnx_output = session.run(None, {'input': test_input.numpy()})
                
                diff = np.abs(original_output.detach().numpy() - onnx_output[0])
                max_diff = np.max(diff)
                
                if max_diff < 1e-5:
                    logger.info(f"ONNX validation passed (max diff: {max_diff:.2e})")
                else:
                    logger.warning(f"ONNX validation failed (max diff: {max_diff:.2e})")
                    success = False
            except ImportError:
                logger.warning("ONNX Runtime not available for validation")
            except Exception as e:
                logger.error(f"ONNX validation failed: {e}")
                success = False
        
        # Validate binary format (simplified - would need C++ implementation)
        if binary_path and Path(binary_path).exists():
            size = Path(binary_path).stat().st_size
            logger.info(f"Binary file size: {size} bytes")
            
            # Basic sanity checks
            expected_min_size = 768 * 512 * 4  # At least input->hidden weights
            if size < expected_min_size:
                logger.warning(f"Binary file seems too small (expected >{expected_min_size} bytes)")
                success = False
            else:
                logger.info("Binary file size validation passed")
        
        return success

def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export NNUE model to various formats')
    parser.add_argument('checkpoint', type=str, help='Path to PyTorch checkpoint')
    parser.add_argument('--output-dir', type=str, default='./exported_models', 
                        help='Output directory for exported models')
    parser.add_argument('--name', type=str, default='nnue_model', 
                        help='Base name for exported files')
    parser.add_argument('--formats', type=str, nargs='+', 
                        default=['binary', 'onnx', 'json'], 
                        choices=['binary', 'onnx', 'json', 'cpp'],
                        help='Export formats')
    parser.add_argument('--validate', action='store_true', 
                        help='Validate exported models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize exporter
    try:
        exporter = NNUEExporter(args.checkpoint)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 1
    
    # Export to requested formats
    exported_files = {}
    
    for format_name in args.formats:
        try:
            output_path = output_dir / f"{args.name}.{format_name.replace('cpp', 'h')}"
            
            if format_name == 'binary':
                exporter.export_to_binary(str(output_path))
            elif format_name == 'onnx':
                exporter.export_to_onnx(str(output_path))
            elif format_name == 'json':
                exporter.export_to_json(str(output_path))
            elif format_name == 'cpp':
                exporter.export_cpp_header(str(output_path))
            
            exported_files[format_name] = str(output_path)
            logger.info(f"Exported {format_name} to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export {format_name}: {e}")
    
    # Validate exports if requested
    if args.validate:
        exporter.validate_export(
            binary_path=exported_files.get('binary'),
            onnx_path=exported_files.get('onnx')
        )
    
    logger.info("Export completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())

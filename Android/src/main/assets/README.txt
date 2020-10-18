RGB
input shape = (416, 416, 3)
range=(0, 1)

===============================
yolov4-416.tflite
-------------------------------
INPUT
[{'name': 'input_1', 'index': 0, 'shape': array([  1, 416, 416,   3], dtype=int32),
'shape_signature': array([  1, 416, 416,   3], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32),
'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
-------------------------------
OUTPUT
BBOX:
[{'name': 'Identity', 'index': 841, 'shape': array([    1, 10647,     4], dtype=int32),
'shape_signature': array([    1, 10647,     4], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0),
'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}},
CONFIDENCE:
{'name': 'Identity_1', 'index': 846, 'shape': array([    1, 10647,    80], dtype=int32),
'shape_signature': array([    1, 10647,    80], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0),
'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}}]

===============================
yolov4-tiny-416.tflite
-------------------------------
INPUT
[{'name': 'input_1', 'index': 0, 'shape': array([  1, 416, 416,   3], dtype=int32),
'shape_signature': array([  1, 416, 416,   3], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}}]
-------------------------------
OUTPUT
CONFIDENCE:
[{'name': 'Identity', 'index': 152, 'shape': array([   1, 2535,   80], dtype=int32),
'shape_signature': array([   1, 2535,   80], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0),
'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}},
BBOX:
{'name': 'Identity_1', 'index': 147, 'shape': array([   1, 2535,    4], dtype=int32),
'shape_signature': array([   1, 2535,    4], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0),
'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}}]

===============================
yolov4full.tflite   FOR ANDROID
-------------------------------
INPUT
[{'name': 'input_1', 'index': 0, 'shape': array([  1, 416, 416,   3], dtype=int32),
'shape_signature': array([  1, 416, 416,   3], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0),
'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}}]
-------------------------------
OUTPUT
[
{'name': 'Identity', 'index': 771, 'shape': array([ 1, 52, 52,  3, 85], dtype=int32),
'shape_signature': array([ 1, 52, 52,  3, 85], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}},
{'name': 'Identity_1', 'index': 802, 'shape': array([ 1, 26, 26,  3, 85], dtype=int32),
'shape_signature': array([ 1, 26, 26,  3, 85], dtype=int32), 'dtype': <class 'numpy.float32'>,
 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
 'sparsity_parameters': {}},
 {'name': 'Identity_2', 'index': 833, 'shape': array([ 1, 13, 13,  3, 85], dtype=int32),
 'shape_signature': array([ 1, 13, 13,  3, 85], dtype=int32), 'dtype': <class 'numpy.float32'>,
 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
 'sparsity_parameters': {}}]

===============================
yolov4full-tiny.tflite   FOR ANDROID
-------------------------------
INPUT
[{'name': 'input_1', 'index': 0, 'shape': array([  1, 416, 416,   3], dtype=int32),
'shape_signature': array([  1, 416, 416,   3], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}}]
-------------------------------
OUTPUT
[
{'name': 'Identity', 'index': 121, 'shape': array([ 1, 13, 13,  3, 85], dtype=int32),
'shape_signature': array([ 1, 13, 13,  3, 85], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}},
{'name': 'Identity_1', 'index': 142, 'shape': array([ 1, 26, 26,  3, 85], dtype=int32),
'shape_signature': array([ 1, 26, 26,  3, 85], dtype=int32), 'dtype': <class 'numpy.float32'>,
'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
'sparsity_parameters': {}}
]



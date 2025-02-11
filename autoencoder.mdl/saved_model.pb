��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/conv2d_5/bias
y
(Adam/v/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/conv2d_5/bias
y
(Adam/m/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_5/kernel
�
*Adam/v/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/kernel*&
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_5/kernel
�
*Adam/m/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/kernel*&
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/v/conv2d_transpose_3/bias
�
2Adam/v/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/m/conv2d_transpose_3/bias
�
2Adam/m/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
�
 Adam/v/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/v/conv2d_transpose_3/kernel
�
4Adam/v/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp Adam/v/conv2d_transpose_3/kernel*&
_output_shapes
:@ *
dtype0
�
 Adam/m/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" Adam/m/conv2d_transpose_3/kernel
�
4Adam/m/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp Adam/m/conv2d_transpose_3/kernel*&
_output_shapes
:@ *
dtype0
�
Adam/v/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/v/conv2d_transpose_2/bias
�
2Adam/v/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_transpose_2/bias*
_output_shapes
: *
dtype0
�
Adam/m/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/m/conv2d_transpose_2/bias
�
2Adam/m/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_transpose_2/bias*
_output_shapes
: *
dtype0
�
 Adam/v/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *1
shared_name" Adam/v/conv2d_transpose_2/kernel
�
4Adam/v/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp Adam/v/conv2d_transpose_2/kernel*&
_output_shapes
:  *
dtype0
�
 Adam/m/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *1
shared_name" Adam/m/conv2d_transpose_2/kernel
�
4Adam/m/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp Adam/m/conv2d_transpose_2/kernel*&
_output_shapes
:  *
dtype0
�
Adam/v/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_4/bias
y
(Adam/v/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/bias*
_output_shapes
: *
dtype0
�
Adam/m/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_4/bias
y
(Adam/m/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/bias*
_output_shapes
: *
dtype0
�
Adam/v/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/v/conv2d_4/kernel
�
*Adam/v/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/kernel*&
_output_shapes
:@ *
dtype0
�
Adam/m/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/m/conv2d_4/kernel
�
*Adam/m/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/kernel*&
_output_shapes
:@ *
dtype0
�
Adam/v/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_3/bias
y
(Adam/v/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/bias*
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_3/bias
y
(Adam/m/conv2d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/bias*
_output_shapes
:@*
dtype0
�
Adam/v/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_3/kernel
�
*Adam/v/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_3/kernel*&
_output_shapes
:@*
dtype0
�
Adam/m/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_3/kernel
�
*Adam/m/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_3/kernel*&
_output_shapes
:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
�
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@*
dtype0
�
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:@*
dtype0
�
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ **
shared_nameconv2d_transpose_3/kernel
�
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:@ *
dtype0
�
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameconv2d_transpose_2/kernel
�
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@ *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_5/kernelconv2d_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_112656

NoOpNoOp
�I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�H
value�HB�H B�H
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder

	optimizer

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
 trace_1
!trace_2
"trace_3* 
* 
�
#layer_with_weights-0
#layer-0
$layer_with_weights-1
$layer-1
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
�
+layer_with_weights-0
+layer-0
,layer_with_weights-1
,layer-1
-layer_with_weights-2
-layer-2
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
�
4
_variables
5_iterations
6_learning_rate
7_index_dict
8
_momentums
9_velocities
:_update_step_xla*

;serving_default* 
OI
VARIABLE_VALUEconv2d_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_transpose_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_transpose_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1*

<0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

kernel
bias
 C_jit_compiled_convolution_op*
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

kernel
bias
 J_jit_compiled_convolution_op*
 
0
1
2
3*
 
0
1
2
3*
* 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

kernel
bias
 ^_jit_compiled_convolution_op*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias
 e_jit_compiled_convolution_op*
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

kernel
bias
 l_jit_compiled_convolution_op*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
6
rtrace_0
strace_1
ttrace_2
utrace_3* 
6
vtrace_0
wtrace_1
xtrace_2
ytrace_3* 
�
50
z1
{2
|3
}4
~5
6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
Q
z0
|1
~2
�3
�4
�5
�6
�7
�8
�9*
Q
{0
}1
2
�3
�4
�5
�6
�7
�8
�9*
* 
* 
<
�	variables
�	keras_api

�total

�count*

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

#0
$1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

+0
,1
-2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
a[
VARIABLE_VALUEAdam/m/conv2d_3/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_3/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_3/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_3/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_4/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_4/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_4/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_4/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/conv2d_transpose_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/conv2d_transpose_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/conv2d_transpose_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/conv2d_transpose_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/conv2d_transpose_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/conv2d_transpose_3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/conv2d_transpose_3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/conv2d_transpose_3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_5/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_5/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_5/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_5/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_5/kernelconv2d_5/bias	iterationlearning_rateAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/biasAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias Adam/m/conv2d_transpose_2/kernel Adam/v/conv2d_transpose_2/kernelAdam/m/conv2d_transpose_2/biasAdam/v/conv2d_transpose_2/bias Adam/m/conv2d_transpose_3/kernel Adam/v/conv2d_transpose_3/kernelAdam/m/conv2d_transpose_3/biasAdam/v/conv2d_transpose_3/biasAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biastotalcountConst*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_113407
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_5/kernelconv2d_5/bias	iterationlearning_rateAdam/m/conv2d_3/kernelAdam/v/conv2d_3/kernelAdam/m/conv2d_3/biasAdam/v/conv2d_3/biasAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/bias Adam/m/conv2d_transpose_2/kernel Adam/v/conv2d_transpose_2/kernelAdam/m/conv2d_transpose_2/biasAdam/v/conv2d_transpose_2/bias Adam/m/conv2d_transpose_3/kernel Adam/v/conv2d_transpose_3/kernelAdam/m/conv2d_transpose_3/biasAdam/v/conv2d_transpose_3/biasAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biastotalcount*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_113519��
�
�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_113074

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������}} i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������}} w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�	
�
-__inference_sequential_2_layer_call_fn_112932

inputs!
unknown:  
	unknown_0: #
	unknown_1:@ 
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112362y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������}} 
 
_user_specified_nameinputs
�!
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_113160

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112304
conv2d_transpose_2_input3
conv2d_transpose_2_112288:  '
conv2d_transpose_2_112290: 3
conv2d_transpose_3_112293:@ '
conv2d_transpose_3_112295:@)
conv2d_5_112298:@
conv2d_5_112300:
identity�� conv2d_5/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputconv2d_transpose_2_112288conv2d_transpose_2_112290*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_112198�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_112293conv2d_transpose_3_112295*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_112243�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_5_112298conv2d_5_112300*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_112278�
IdentityIdentity)conv2d_5/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_5/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:i e
/
_output_shapes
:���������}} 
2
_user_specified_nameconv2d_transpose_2_input
�	
�
-__inference_sequential_2_layer_call_fn_112915

inputs!
unknown:  
	unknown_0: #
	unknown_1:@ 
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112326y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������}} 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_113054

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�!
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_112243

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_112278

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
)__inference_conv2d_3_layer_call_fn_113043

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_112024y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_4_layer_call_fn_113063

inputs!
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_112041w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�

�
(__inference_denoise_layer_call_fn_112706
x!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:@ 
	unknown_6:@#
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_denoise_layer_call_and_return_conditional_losses_112554y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:�����������

_user_specified_namex
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112326

inputs3
conv2d_transpose_2_112310:  '
conv2d_transpose_2_112312: 3
conv2d_transpose_3_112315:@ '
conv2d_transpose_3_112317:@)
conv2d_5_112320:@
conv2d_5_112322:
identity�� conv2d_5/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_112310conv2d_transpose_2_112312*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_112198�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_112315conv2d_transpose_3_112317*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_112243�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_5_112320conv2d_5_112322*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_112278�
IdentityIdentity)conv2d_5/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_5/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������}} 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_112024

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112106

inputs)
conv2d_3_112095:@
conv2d_3_112097:@)
conv2d_4_112100:@ 
conv2d_4_112102: 
identity�� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_112095conv2d_3_112097*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_112024�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_112100conv2d_4_112102*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_112041�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} �
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112048
input_2)
conv2d_3_112025:@
conv2d_3_112027:@)
conv2d_4_112042:@ 
conv2d_4_112044: 
identity�� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_3_112025conv2d_3_112027*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_112024�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_112042conv2d_4_112044*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_112041�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} �
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
�
-__inference_sequential_1_layer_call_fn_112862

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_3_layer_call_fn_113126

inputs!
unknown:@ 
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_112243�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_112849

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112079w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�e
�
C__inference_denoise_layer_call_and_return_conditional_losses_112836
xN
4sequential_1_conv2d_3_conv2d_readvariableop_resource:@C
5sequential_1_conv2d_3_biasadd_readvariableop_resource:@N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:@ C
5sequential_1_conv2d_4_biasadd_readvariableop_resource: b
Hsequential_2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:  M
?sequential_2_conv2d_transpose_2_biasadd_readvariableop_resource: b
Hsequential_2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ M
?sequential_2_conv2d_transpose_3_biasadd_readvariableop_resource:@N
4sequential_2_conv2d_5_conv2d_readvariableop_resource:@C
5sequential_2_conv2d_5_biasadd_readvariableop_resource:
identity��,sequential_1/conv2d_3/BiasAdd/ReadVariableOp�+sequential_1/conv2d_3/Conv2D/ReadVariableOp�,sequential_1/conv2d_4/BiasAdd/ReadVariableOp�+sequential_1/conv2d_4/Conv2D/ReadVariableOp�,sequential_2/conv2d_5/BiasAdd/ReadVariableOp�+sequential_2/conv2d_5/Conv2D/ReadVariableOp�6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp�?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp�?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential_1/conv2d_3/Conv2DConv2Dx3sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
sequential_1/conv2d_4/Conv2DConv2D(sequential_1/conv2d_3/Relu:activations:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
�
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} �
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������}} �
%sequential_2/conv2d_transpose_2/ShapeShape(sequential_1/conv2d_4/Relu:activations:0*
T0*
_output_shapes
::��}
3sequential_2/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential_2/conv2d_transpose_2/strided_sliceStridedSlice.sequential_2/conv2d_transpose_2/Shape:output:0<sequential_2/conv2d_transpose_2/strided_slice/stack:output:0>sequential_2/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_2/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'sequential_2/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�j
'sequential_2/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�i
'sequential_2/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
%sequential_2/conv2d_transpose_2/stackPack6sequential_2/conv2d_transpose_2/strided_slice:output:00sequential_2/conv2d_transpose_2/stack/1:output:00sequential_2/conv2d_transpose_2/stack/2:output:00sequential_2/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_2/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_2/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_2/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_2/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_2/conv2d_transpose_2/stack:output:0>sequential_2/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_2/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_2/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
0sequential_2/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_2/conv2d_transpose_2/stack:output:0Gsequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0(sequential_1/conv2d_4/Relu:activations:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_2_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
'sequential_2/conv2d_transpose_2/BiasAddBiasAdd9sequential_2/conv2d_transpose_2/conv2d_transpose:output:0>sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
$sequential_2/conv2d_transpose_2/ReluRelu0sequential_2/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
%sequential_2/conv2d_transpose_3/ShapeShape2sequential_2/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
::��}
3sequential_2/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential_2/conv2d_transpose_3/strided_sliceStridedSlice.sequential_2/conv2d_transpose_3/Shape:output:0<sequential_2/conv2d_transpose_3/strided_slice/stack:output:0>sequential_2/conv2d_transpose_3/strided_slice/stack_1:output:0>sequential_2/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'sequential_2/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�j
'sequential_2/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�i
'sequential_2/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
%sequential_2/conv2d_transpose_3/stackPack6sequential_2/conv2d_transpose_3/strided_slice:output:00sequential_2/conv2d_transpose_3/stack/1:output:00sequential_2/conv2d_transpose_3/stack/2:output:00sequential_2/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_2/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_2/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_2/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_2/conv2d_transpose_3/strided_slice_1StridedSlice.sequential_2/conv2d_transpose_3/stack:output:0>sequential_2/conv2d_transpose_3/strided_slice_1/stack:output:0@sequential_2/conv2d_transpose_3/strided_slice_1/stack_1:output:0@sequential_2/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
0sequential_2/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.sequential_2/conv2d_transpose_3/stack:output:0Gsequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:02sequential_2/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_2_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'sequential_2/conv2d_transpose_3/BiasAddBiasAdd9sequential_2/conv2d_transpose_3/conv2d_transpose:output:0>sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
$sequential_2/conv2d_transpose_3/ReluRelu0sequential_2/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
+sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential_2/conv2d_5/Conv2DConv2D2sequential_2/conv2d_transpose_3/Relu:activations:03sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
,sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_5/BiasAddBiasAdd%sequential_2/conv2d_5/Conv2D:output:04sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential_2/conv2d_5/SigmoidSigmoid&sequential_2/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������z
IdentityIdentity!sequential_2/conv2d_5/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_2/conv2d_5/BiasAdd/ReadVariableOp,^sequential_2/conv2d_5/Conv2D/ReadVariableOp7^sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp@^sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp,sequential_2/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_5/Conv2D/ReadVariableOp+sequential_2/conv2d_5/Conv2D/ReadVariableOp2p
6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp2�
?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp2�
?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:T P
1
_output_shapes
:�����������

_user_specified_namex
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112898

inputsA
'conv2d_3_conv2d_readvariableop_resource:@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@ 6
(conv2d_4_biasadd_readvariableop_resource: 
identity��conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@l
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������}} r
IdentityIdentityconv2d_4/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������}} �
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_112656
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:@ 
	unknown_6:@#
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_112009y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�o
�
!__inference__wrapped_model_112009
input_1V
<denoise_sequential_1_conv2d_3_conv2d_readvariableop_resource:@K
=denoise_sequential_1_conv2d_3_biasadd_readvariableop_resource:@V
<denoise_sequential_1_conv2d_4_conv2d_readvariableop_resource:@ K
=denoise_sequential_1_conv2d_4_biasadd_readvariableop_resource: j
Pdenoise_sequential_2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:  U
Gdenoise_sequential_2_conv2d_transpose_2_biasadd_readvariableop_resource: j
Pdenoise_sequential_2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ U
Gdenoise_sequential_2_conv2d_transpose_3_biasadd_readvariableop_resource:@V
<denoise_sequential_2_conv2d_5_conv2d_readvariableop_resource:@K
=denoise_sequential_2_conv2d_5_biasadd_readvariableop_resource:
identity��4denoise/sequential_1/conv2d_3/BiasAdd/ReadVariableOp�3denoise/sequential_1/conv2d_3/Conv2D/ReadVariableOp�4denoise/sequential_1/conv2d_4/BiasAdd/ReadVariableOp�3denoise/sequential_1/conv2d_4/Conv2D/ReadVariableOp�4denoise/sequential_2/conv2d_5/BiasAdd/ReadVariableOp�3denoise/sequential_2/conv2d_5/Conv2D/ReadVariableOp�>denoise/sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp�Gdenoise/sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�>denoise/sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp�Gdenoise/sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
3denoise/sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp<denoise_sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
$denoise/sequential_1/conv2d_3/Conv2DConv2Dinput_1;denoise/sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
4denoise/sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp=denoise_sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%denoise/sequential_1/conv2d_3/BiasAddBiasAdd-denoise/sequential_1/conv2d_3/Conv2D:output:0<denoise/sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
"denoise/sequential_1/conv2d_3/ReluRelu.denoise/sequential_1/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
3denoise/sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<denoise_sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
$denoise/sequential_1/conv2d_4/Conv2DConv2D0denoise/sequential_1/conv2d_3/Relu:activations:0;denoise/sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
�
4denoise/sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=denoise_sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
%denoise/sequential_1/conv2d_4/BiasAddBiasAdd-denoise/sequential_1/conv2d_4/Conv2D:output:0<denoise/sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} �
"denoise/sequential_1/conv2d_4/ReluRelu.denoise/sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������}} �
-denoise/sequential_2/conv2d_transpose_2/ShapeShape0denoise/sequential_1/conv2d_4/Relu:activations:0*
T0*
_output_shapes
::���
;denoise/sequential_2/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=denoise/sequential_2/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=denoise/sequential_2/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5denoise/sequential_2/conv2d_transpose_2/strided_sliceStridedSlice6denoise/sequential_2/conv2d_transpose_2/Shape:output:0Ddenoise/sequential_2/conv2d_transpose_2/strided_slice/stack:output:0Fdenoise/sequential_2/conv2d_transpose_2/strided_slice/stack_1:output:0Fdenoise/sequential_2/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
/denoise/sequential_2/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�r
/denoise/sequential_2/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�q
/denoise/sequential_2/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
-denoise/sequential_2/conv2d_transpose_2/stackPack>denoise/sequential_2/conv2d_transpose_2/strided_slice:output:08denoise/sequential_2/conv2d_transpose_2/stack/1:output:08denoise/sequential_2/conv2d_transpose_2/stack/2:output:08denoise/sequential_2/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:�
=denoise/sequential_2/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
?denoise/sequential_2/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?denoise/sequential_2/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7denoise/sequential_2/conv2d_transpose_2/strided_slice_1StridedSlice6denoise/sequential_2/conv2d_transpose_2/stack:output:0Fdenoise/sequential_2/conv2d_transpose_2/strided_slice_1/stack:output:0Hdenoise/sequential_2/conv2d_transpose_2/strided_slice_1/stack_1:output:0Hdenoise/sequential_2/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Gdenoise/sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpPdenoise_sequential_2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
8denoise/sequential_2/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput6denoise/sequential_2/conv2d_transpose_2/stack:output:0Odenoise/sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:00denoise/sequential_1/conv2d_4/Relu:activations:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
>denoise/sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpGdenoise_sequential_2_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
/denoise/sequential_2/conv2d_transpose_2/BiasAddBiasAddAdenoise/sequential_2/conv2d_transpose_2/conv2d_transpose:output:0Fdenoise/sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
,denoise/sequential_2/conv2d_transpose_2/ReluRelu8denoise/sequential_2/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
-denoise/sequential_2/conv2d_transpose_3/ShapeShape:denoise/sequential_2/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
::���
;denoise/sequential_2/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
=denoise/sequential_2/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
=denoise/sequential_2/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
5denoise/sequential_2/conv2d_transpose_3/strided_sliceStridedSlice6denoise/sequential_2/conv2d_transpose_3/Shape:output:0Ddenoise/sequential_2/conv2d_transpose_3/strided_slice/stack:output:0Fdenoise/sequential_2/conv2d_transpose_3/strided_slice/stack_1:output:0Fdenoise/sequential_2/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
/denoise/sequential_2/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�r
/denoise/sequential_2/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�q
/denoise/sequential_2/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
-denoise/sequential_2/conv2d_transpose_3/stackPack>denoise/sequential_2/conv2d_transpose_3/strided_slice:output:08denoise/sequential_2/conv2d_transpose_3/stack/1:output:08denoise/sequential_2/conv2d_transpose_3/stack/2:output:08denoise/sequential_2/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:�
=denoise/sequential_2/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
?denoise/sequential_2/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
?denoise/sequential_2/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
7denoise/sequential_2/conv2d_transpose_3/strided_slice_1StridedSlice6denoise/sequential_2/conv2d_transpose_3/stack:output:0Fdenoise/sequential_2/conv2d_transpose_3/strided_slice_1/stack:output:0Hdenoise/sequential_2/conv2d_transpose_3/strided_slice_1/stack_1:output:0Hdenoise/sequential_2/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Gdenoise/sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpPdenoise_sequential_2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
8denoise/sequential_2/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput6denoise/sequential_2/conv2d_transpose_3/stack:output:0Odenoise/sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0:denoise/sequential_2/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
>denoise/sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpGdenoise_sequential_2_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
/denoise/sequential_2/conv2d_transpose_3/BiasAddBiasAddAdenoise/sequential_2/conv2d_transpose_3/conv2d_transpose:output:0Fdenoise/sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
,denoise/sequential_2/conv2d_transpose_3/ReluRelu8denoise/sequential_2/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
3denoise/sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp<denoise_sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
$denoise/sequential_2/conv2d_5/Conv2DConv2D:denoise/sequential_2/conv2d_transpose_3/Relu:activations:0;denoise/sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
4denoise/sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp=denoise_sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%denoise/sequential_2/conv2d_5/BiasAddBiasAdd-denoise/sequential_2/conv2d_5/Conv2D:output:0<denoise/sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
%denoise/sequential_2/conv2d_5/SigmoidSigmoid.denoise/sequential_2/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:������������
IdentityIdentity)denoise/sequential_2/conv2d_5/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp5^denoise/sequential_1/conv2d_3/BiasAdd/ReadVariableOp4^denoise/sequential_1/conv2d_3/Conv2D/ReadVariableOp5^denoise/sequential_1/conv2d_4/BiasAdd/ReadVariableOp4^denoise/sequential_1/conv2d_4/Conv2D/ReadVariableOp5^denoise/sequential_2/conv2d_5/BiasAdd/ReadVariableOp4^denoise/sequential_2/conv2d_5/Conv2D/ReadVariableOp?^denoise/sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOpH^denoise/sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?^denoise/sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOpH^denoise/sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2l
4denoise/sequential_1/conv2d_3/BiasAdd/ReadVariableOp4denoise/sequential_1/conv2d_3/BiasAdd/ReadVariableOp2j
3denoise/sequential_1/conv2d_3/Conv2D/ReadVariableOp3denoise/sequential_1/conv2d_3/Conv2D/ReadVariableOp2l
4denoise/sequential_1/conv2d_4/BiasAdd/ReadVariableOp4denoise/sequential_1/conv2d_4/BiasAdd/ReadVariableOp2j
3denoise/sequential_1/conv2d_4/Conv2D/ReadVariableOp3denoise/sequential_1/conv2d_4/Conv2D/ReadVariableOp2l
4denoise/sequential_2/conv2d_5/BiasAdd/ReadVariableOp4denoise/sequential_2/conv2d_5/BiasAdd/ReadVariableOp2j
3denoise/sequential_2/conv2d_5/Conv2D/ReadVariableOp3denoise/sequential_2/conv2d_5/Conv2D/ReadVariableOp2�
>denoise/sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp>denoise/sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp2�
Gdenoise/sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpGdenoise/sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2�
>denoise/sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp>denoise/sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp2�
Gdenoise/sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpGdenoise/sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
)__inference_conv2d_5_layer_call_fn_113169

inputs!
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_112278y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
C__inference_denoise_layer_call_and_return_conditional_losses_112474
input_1-
sequential_1_112451:@!
sequential_1_112453:@-
sequential_1_112455:@ !
sequential_1_112457: -
sequential_2_112460:  !
sequential_2_112462: -
sequential_2_112464:@ !
sequential_2_112466:@-
sequential_2_112468:@!
sequential_2_112470:
identity��$sequential_1/StatefulPartitionedCall�$sequential_2/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_112451sequential_1_112453sequential_1_112455sequential_1_112457*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112106�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_112460sequential_2_112462sequential_2_112464sequential_2_112466sequential_2_112468sequential_2_112470*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112362�
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112285
conv2d_transpose_2_input3
conv2d_transpose_2_112257:  '
conv2d_transpose_2_112259: 3
conv2d_transpose_3_112262:@ '
conv2d_transpose_3_112264:@)
conv2d_5_112279:@
conv2d_5_112281:
identity�� conv2d_5/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputconv2d_transpose_2_112257conv2d_transpose_2_112259*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_112198�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_112262conv2d_transpose_3_112264*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_112243�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_5_112279conv2d_5_112281*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_112278�
IdentityIdentity)conv2d_5/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_5/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:i e
/
_output_shapes
:���������}} 
2
_user_specified_nameconv2d_transpose_2_input
�
�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_112041

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������}} i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������}} w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
C__inference_denoise_layer_call_and_return_conditional_losses_112448
input_1-
sequential_1_112425:@!
sequential_1_112427:@-
sequential_1_112429:@ !
sequential_1_112431: -
sequential_2_112434:  !
sequential_2_112436: -
sequential_2_112438:@ !
sequential_2_112440:@-
sequential_2_112442:@!
sequential_2_112444:
identity��$sequential_1/StatefulPartitionedCall�$sequential_2/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_112425sequential_1_112427sequential_1_112429sequential_1_112431*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112079�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_112434sequential_2_112436sequential_2_112438sequential_2_112440sequential_2_112442sequential_2_112444*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112326�
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�e
�
C__inference_denoise_layer_call_and_return_conditional_losses_112771
xN
4sequential_1_conv2d_3_conv2d_readvariableop_resource:@C
5sequential_1_conv2d_3_biasadd_readvariableop_resource:@N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:@ C
5sequential_1_conv2d_4_biasadd_readvariableop_resource: b
Hsequential_2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:  M
?sequential_2_conv2d_transpose_2_biasadd_readvariableop_resource: b
Hsequential_2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ M
?sequential_2_conv2d_transpose_3_biasadd_readvariableop_resource:@N
4sequential_2_conv2d_5_conv2d_readvariableop_resource:@C
5sequential_2_conv2d_5_biasadd_readvariableop_resource:
identity��,sequential_1/conv2d_3/BiasAdd/ReadVariableOp�+sequential_1/conv2d_3/Conv2D/ReadVariableOp�,sequential_1/conv2d_4/BiasAdd/ReadVariableOp�+sequential_1/conv2d_4/Conv2D/ReadVariableOp�,sequential_2/conv2d_5/BiasAdd/ReadVariableOp�+sequential_2/conv2d_5/Conv2D/ReadVariableOp�6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp�?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp�?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential_1/conv2d_3/Conv2DConv2Dx3sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
sequential_1/conv2d_4/Conv2DConv2D(sequential_1/conv2d_3/Relu:activations:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
�
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} �
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������}} �
%sequential_2/conv2d_transpose_2/ShapeShape(sequential_1/conv2d_4/Relu:activations:0*
T0*
_output_shapes
::��}
3sequential_2/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential_2/conv2d_transpose_2/strided_sliceStridedSlice.sequential_2/conv2d_transpose_2/Shape:output:0<sequential_2/conv2d_transpose_2/strided_slice/stack:output:0>sequential_2/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_2/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'sequential_2/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�j
'sequential_2/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�i
'sequential_2/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
%sequential_2/conv2d_transpose_2/stackPack6sequential_2/conv2d_transpose_2/strided_slice:output:00sequential_2/conv2d_transpose_2/stack/1:output:00sequential_2/conv2d_transpose_2/stack/2:output:00sequential_2/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_2/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_2/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_2/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_2/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_2/conv2d_transpose_2/stack:output:0>sequential_2/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_2/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_2/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
0sequential_2/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_2/conv2d_transpose_2/stack:output:0Gsequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0(sequential_1/conv2d_4/Relu:activations:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_2_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
'sequential_2/conv2d_transpose_2/BiasAddBiasAdd9sequential_2/conv2d_transpose_2/conv2d_transpose:output:0>sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
$sequential_2/conv2d_transpose_2/ReluRelu0sequential_2/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
%sequential_2/conv2d_transpose_3/ShapeShape2sequential_2/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
::��}
3sequential_2/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential_2/conv2d_transpose_3/strided_sliceStridedSlice.sequential_2/conv2d_transpose_3/Shape:output:0<sequential_2/conv2d_transpose_3/strided_slice/stack:output:0>sequential_2/conv2d_transpose_3/strided_slice/stack_1:output:0>sequential_2/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'sequential_2/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�j
'sequential_2/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�i
'sequential_2/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
%sequential_2/conv2d_transpose_3/stackPack6sequential_2/conv2d_transpose_3/strided_slice:output:00sequential_2/conv2d_transpose_3/stack/1:output:00sequential_2/conv2d_transpose_3/stack/2:output:00sequential_2/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_2/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
7sequential_2/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
7sequential_2/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
/sequential_2/conv2d_transpose_3/strided_slice_1StridedSlice.sequential_2/conv2d_transpose_3/stack:output:0>sequential_2/conv2d_transpose_3/strided_slice_1/stack:output:0@sequential_2/conv2d_transpose_3/strided_slice_1/stack_1:output:0@sequential_2/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
0sequential_2/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.sequential_2/conv2d_transpose_3/stack:output:0Gsequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:02sequential_2/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_2_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
'sequential_2/conv2d_transpose_3/BiasAddBiasAdd9sequential_2/conv2d_transpose_3/conv2d_transpose:output:0>sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
$sequential_2/conv2d_transpose_3/ReluRelu0sequential_2/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
+sequential_2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential_2/conv2d_5/Conv2DConv2D2sequential_2/conv2d_transpose_3/Relu:activations:03sequential_2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
,sequential_2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_5/BiasAddBiasAdd%sequential_2/conv2d_5/Conv2D:output:04sequential_2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential_2/conv2d_5/SigmoidSigmoid&sequential_2/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������z
IdentityIdentity!sequential_2/conv2d_5/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_2/conv2d_5/BiasAdd/ReadVariableOp,^sequential_2/conv2d_5/Conv2D/ReadVariableOp7^sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp@^sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_5/BiasAdd/ReadVariableOp,sequential_2/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_5/Conv2D/ReadVariableOp+sequential_2/conv2d_5/Conv2D/ReadVariableOp2p
6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_2/conv2d_transpose_2/BiasAdd/ReadVariableOp2�
?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp6sequential_2/conv2d_transpose_3/BiasAdd/ReadVariableOp2�
?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?sequential_2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:T P
1
_output_shapes
:�����������

_user_specified_namex
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112062
input_2)
conv2d_3_112051:@
conv2d_3_112053:@)
conv2d_4_112056:@ 
conv2d_4_112058: 
identity�� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_3_112051conv2d_3_112053*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_112024�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_112056conv2d_4_112058*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_112041�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} �
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
̑
�
"__inference__traced_restore_113519
file_prefix:
 assignvariableop_conv2d_3_kernel:@.
 assignvariableop_1_conv2d_3_bias:@<
"assignvariableop_2_conv2d_4_kernel:@ .
 assignvariableop_3_conv2d_4_bias: F
,assignvariableop_4_conv2d_transpose_2_kernel:  8
*assignvariableop_5_conv2d_transpose_2_bias: F
,assignvariableop_6_conv2d_transpose_3_kernel:@ 8
*assignvariableop_7_conv2d_transpose_3_bias:@<
"assignvariableop_8_conv2d_5_kernel:@.
 assignvariableop_9_conv2d_5_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: D
*assignvariableop_12_adam_m_conv2d_3_kernel:@D
*assignvariableop_13_adam_v_conv2d_3_kernel:@6
(assignvariableop_14_adam_m_conv2d_3_bias:@6
(assignvariableop_15_adam_v_conv2d_3_bias:@D
*assignvariableop_16_adam_m_conv2d_4_kernel:@ D
*assignvariableop_17_adam_v_conv2d_4_kernel:@ 6
(assignvariableop_18_adam_m_conv2d_4_bias: 6
(assignvariableop_19_adam_v_conv2d_4_bias: N
4assignvariableop_20_adam_m_conv2d_transpose_2_kernel:  N
4assignvariableop_21_adam_v_conv2d_transpose_2_kernel:  @
2assignvariableop_22_adam_m_conv2d_transpose_2_bias: @
2assignvariableop_23_adam_v_conv2d_transpose_2_bias: N
4assignvariableop_24_adam_m_conv2d_transpose_3_kernel:@ N
4assignvariableop_25_adam_v_conv2d_transpose_3_kernel:@ @
2assignvariableop_26_adam_m_conv2d_transpose_3_bias:@@
2assignvariableop_27_adam_v_conv2d_transpose_3_bias:@D
*assignvariableop_28_adam_m_conv2d_5_kernel:@D
*assignvariableop_29_adam_v_conv2d_5_kernel:@6
(assignvariableop_30_adam_m_conv2d_5_bias:6
(assignvariableop_31_adam_v_conv2d_5_bias:#
assignvariableop_32_total: #
assignvariableop_33_count: 
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_5_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_5_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_m_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_v_conv2d_3_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_m_conv2d_3_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_v_conv2d_3_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_m_conv2d_4_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_v_conv2d_4_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_m_conv2d_4_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_v_conv2d_4_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_m_conv2d_transpose_2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_v_conv2d_transpose_2_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_m_conv2d_transpose_2_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_v_conv2d_transpose_2_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_m_conv2d_transpose_3_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_v_conv2d_transpose_3_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_m_conv2d_transpose_3_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_v_conv2d_transpose_3_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_m_conv2d_5_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_v_conv2d_5_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_m_conv2d_5_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_v_conv2d_5_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
� 
__inference__traced_save_113407
file_prefix@
&read_disablecopyonread_conv2d_3_kernel:@4
&read_1_disablecopyonread_conv2d_3_bias:@B
(read_2_disablecopyonread_conv2d_4_kernel:@ 4
&read_3_disablecopyonread_conv2d_4_bias: L
2read_4_disablecopyonread_conv2d_transpose_2_kernel:  >
0read_5_disablecopyonread_conv2d_transpose_2_bias: L
2read_6_disablecopyonread_conv2d_transpose_3_kernel:@ >
0read_7_disablecopyonread_conv2d_transpose_3_bias:@B
(read_8_disablecopyonread_conv2d_5_kernel:@4
&read_9_disablecopyonread_conv2d_5_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: J
0read_12_disablecopyonread_adam_m_conv2d_3_kernel:@J
0read_13_disablecopyonread_adam_v_conv2d_3_kernel:@<
.read_14_disablecopyonread_adam_m_conv2d_3_bias:@<
.read_15_disablecopyonread_adam_v_conv2d_3_bias:@J
0read_16_disablecopyonread_adam_m_conv2d_4_kernel:@ J
0read_17_disablecopyonread_adam_v_conv2d_4_kernel:@ <
.read_18_disablecopyonread_adam_m_conv2d_4_bias: <
.read_19_disablecopyonread_adam_v_conv2d_4_bias: T
:read_20_disablecopyonread_adam_m_conv2d_transpose_2_kernel:  T
:read_21_disablecopyonread_adam_v_conv2d_transpose_2_kernel:  F
8read_22_disablecopyonread_adam_m_conv2d_transpose_2_bias: F
8read_23_disablecopyonread_adam_v_conv2d_transpose_2_bias: T
:read_24_disablecopyonread_adam_m_conv2d_transpose_3_kernel:@ T
:read_25_disablecopyonread_adam_v_conv2d_transpose_3_kernel:@ F
8read_26_disablecopyonread_adam_m_conv2d_transpose_3_bias:@F
8read_27_disablecopyonread_adam_v_conv2d_transpose_3_bias:@J
0read_28_disablecopyonread_adam_m_conv2d_5_kernel:@J
0read_29_disablecopyonread_adam_v_conv2d_5_kernel:@<
.read_30_disablecopyonread_adam_m_conv2d_5_bias:<
.read_31_disablecopyonread_adam_v_conv2d_5_bias:)
read_32_disablecopyonread_total: )
read_33_disablecopyonread_count: 
savev2_const
identity_69��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_3_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_3_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_4/DisableCopyOnReadDisableCopyOnRead2read_4_disablecopyonread_conv2d_transpose_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp2read_4_disablecopyonread_conv2d_transpose_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_5/DisableCopyOnReadDisableCopyOnRead0read_5_disablecopyonread_conv2d_transpose_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp0read_5_disablecopyonread_conv2d_transpose_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnRead2read_6_disablecopyonread_conv2d_transpose_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp2read_6_disablecopyonread_conv2d_transpose_3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_7/DisableCopyOnReadDisableCopyOnRead0read_7_disablecopyonread_conv2d_transpose_3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp0read_7_disablecopyonread_conv2d_transpose_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_conv2d_5_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:@z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_conv2d_5_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_adam_m_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_adam_m_conv2d_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_v_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_v_conv2d_3_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_14/DisableCopyOnReadDisableCopyOnRead.read_14_disablecopyonread_adam_m_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp.read_14_disablecopyonread_adam_m_conv2d_3_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_15/DisableCopyOnReadDisableCopyOnRead.read_15_disablecopyonread_adam_v_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp.read_15_disablecopyonread_adam_v_conv2d_3_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_m_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_m_conv2d_4_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_v_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_v_conv2d_4_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_m_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_m_conv2d_4_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_adam_v_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_adam_v_conv2d_4_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead:read_20_disablecopyonread_adam_m_conv2d_transpose_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp:read_20_disablecopyonread_adam_m_conv2d_transpose_2_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_21/DisableCopyOnReadDisableCopyOnRead:read_21_disablecopyonread_adam_v_conv2d_transpose_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp:read_21_disablecopyonread_adam_v_conv2d_transpose_2_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_22/DisableCopyOnReadDisableCopyOnRead8read_22_disablecopyonread_adam_m_conv2d_transpose_2_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp8read_22_disablecopyonread_adam_m_conv2d_transpose_2_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_23/DisableCopyOnReadDisableCopyOnRead8read_23_disablecopyonread_adam_v_conv2d_transpose_2_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp8read_23_disablecopyonread_adam_v_conv2d_transpose_2_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnRead:read_24_disablecopyonread_adam_m_conv2d_transpose_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp:read_24_disablecopyonread_adam_m_conv2d_transpose_3_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_25/DisableCopyOnReadDisableCopyOnRead:read_25_disablecopyonread_adam_v_conv2d_transpose_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp:read_25_disablecopyonread_adam_v_conv2d_transpose_3_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_26/DisableCopyOnReadDisableCopyOnRead8read_26_disablecopyonread_adam_m_conv2d_transpose_3_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp8read_26_disablecopyonread_adam_m_conv2d_transpose_3_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_27/DisableCopyOnReadDisableCopyOnRead8read_27_disablecopyonread_adam_v_conv2d_transpose_3_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp8read_27_disablecopyonread_adam_v_conv2d_transpose_3_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_28/DisableCopyOnReadDisableCopyOnRead0read_28_disablecopyonread_adam_m_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp0read_28_disablecopyonread_adam_m_conv2d_5_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnRead0read_29_disablecopyonread_adam_v_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp0read_29_disablecopyonread_adam_v_conv2d_5_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_m_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_m_conv2d_5_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnRead.read_31_disablecopyonread_adam_v_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp.read_31_disablecopyonread_adam_v_conv2d_5_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_32/DisableCopyOnReadDisableCopyOnReadread_32_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpread_32_disablecopyonread_total^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_33/DisableCopyOnReadDisableCopyOnReadread_33_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpread_33_disablecopyonread_count^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *1
dtypes'
%2#	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_68Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_69IdentityIdentity_68:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_69Identity_69:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:#

_output_shapes
: 
�

�
(__inference_denoise_layer_call_fn_112681
x!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:@ 
	unknown_6:@#
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_denoise_layer_call_and_return_conditional_losses_112503y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:�����������

_user_specified_namex
�B
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112983

inputsU
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:  @
2conv2d_transpose_2_biasadd_readvariableop_resource: U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ @
2conv2d_transpose_3_biasadd_readvariableop_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@6
(conv2d_5_biasadd_readvariableop_resource:
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�)conv2d_transpose_2/BiasAdd/ReadVariableOp�2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp\
conv2d_transpose_2/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� {
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_5/Conv2DConv2D%conv2d_transpose_3/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������r
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������m
IdentityIdentityconv2d_5/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������}} 
 
_user_specified_nameinputs
�!
�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_113117

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112079

inputs)
conv2d_3_112068:@
conv2d_3_112070:@)
conv2d_4_112073:@ 
conv2d_4_112075: 
identity�� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_112068conv2d_3_112070*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_112024�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_112073conv2d_4_112075*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_112041�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} �
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112362

inputs3
conv2d_transpose_2_112346:  '
conv2d_transpose_2_112348: 3
conv2d_transpose_3_112351:@ '
conv2d_transpose_3_112353:@)
conv2d_5_112356:@
conv2d_5_112358:
identity�� conv2d_5/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_112346conv2d_transpose_2_112348*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_112198�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_112351conv2d_transpose_3_112353*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_112243�
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_5_112356conv2d_5_112358*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_112278�
IdentityIdentity)conv2d_5/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_5/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:���������}} 
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_112090
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112079w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�
�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_113180

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�B
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_113034

inputsU
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:  @
2conv2d_transpose_2_biasadd_readvariableop_resource: U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@ @
2conv2d_transpose_3_biasadd_readvariableop_resource:@A
'conv2d_5_conv2d_readvariableop_resource:@6
(conv2d_5_biasadd_readvariableop_resource:
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�)conv2d_transpose_2/BiasAdd/ReadVariableOp�2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�)conv2d_transpose_3/BiasAdd/ReadVariableOp�2conv2d_transpose_3/conv2d_transpose/ReadVariableOp\
conv2d_transpose_2/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� {
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
::��p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_5/Conv2DConv2D%conv2d_transpose_3/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������r
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������m
IdentityIdentityconv2d_5/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������}} 
 
_user_specified_nameinputs
�
�
C__inference_denoise_layer_call_and_return_conditional_losses_112503
x-
sequential_1_112480:@!
sequential_1_112482:@-
sequential_1_112484:@ !
sequential_1_112486: -
sequential_2_112489:  !
sequential_2_112491: -
sequential_2_112493:@ !
sequential_2_112495:@-
sequential_2_112497:@!
sequential_2_112499:
identity��$sequential_1/StatefulPartitionedCall�$sequential_2/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallxsequential_1_112480sequential_1_112482sequential_1_112484sequential_1_112486*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112079�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_112489sequential_2_112491sequential_2_112493sequential_2_112495sequential_2_112497sequential_2_112499*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112326�
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:T P
1
_output_shapes
:�����������

_user_specified_namex
�	
�
-__inference_sequential_2_layer_call_fn_112341
conv2d_transpose_2_input!
unknown:  
	unknown_0: #
	unknown_1:@ 
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112326y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:���������}} 
2
_user_specified_nameconv2d_transpose_2_input
�!
�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_112198

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_2_layer_call_fn_113083

inputs!
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_112198�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112880

inputsA
'conv2d_3_conv2d_readvariableop_resource:@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@ 6
(conv2d_4_biasadd_readvariableop_resource: 
identity��conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@l
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_4/Conv2DConv2Dconv2d_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} *
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������}} j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������}} r
IdentityIdentityconv2d_4/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������}} �
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_2_layer_call_fn_112377
conv2d_transpose_2_input!
unknown:  
	unknown_0: #
	unknown_1:@ 
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112362y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������}} : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
/
_output_shapes
:���������}} 
2
_user_specified_nameconv2d_transpose_2_input
�
�
C__inference_denoise_layer_call_and_return_conditional_losses_112554
x-
sequential_1_112531:@!
sequential_1_112533:@-
sequential_1_112535:@ !
sequential_1_112537: -
sequential_2_112540:  !
sequential_2_112542: -
sequential_2_112544:@ !
sequential_2_112546:@-
sequential_2_112548:@!
sequential_2_112550:
identity��$sequential_1/StatefulPartitionedCall�$sequential_2/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallxsequential_1_112531sequential_1_112533sequential_1_112535sequential_1_112537*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112106�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0sequential_2_112540sequential_2_112542sequential_2_112544sequential_2_112546sequential_2_112548sequential_2_112550*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_112362�
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:T P
1
_output_shapes
:�����������

_user_specified_namex
�

�
(__inference_denoise_layer_call_fn_112577
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:@ 
	unknown_6:@#
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_denoise_layer_call_and_return_conditional_losses_112554y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
-__inference_sequential_1_layer_call_fn_112117
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������}} *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_112106w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������}} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_2
�

�
(__inference_denoise_layer_call_fn_112526
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:@ 
	unknown_6:@#
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_denoise_layer_call_and_return_conditional_losses_112503y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:�����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������F
output_1:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder

	optimizer

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
(__inference_denoise_layer_call_fn_112526
(__inference_denoise_layer_call_fn_112577
(__inference_denoise_layer_call_fn_112681
(__inference_denoise_layer_call_fn_112706�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
 trace_1
!trace_2
"trace_32�
C__inference_denoise_layer_call_and_return_conditional_losses_112448
C__inference_denoise_layer_call_and_return_conditional_losses_112474
C__inference_denoise_layer_call_and_return_conditional_losses_112771
C__inference_denoise_layer_call_and_return_conditional_losses_112836�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0z trace_1z!trace_2z"trace_3
�B�
!__inference__wrapped_model_112009input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
#layer_with_weights-0
#layer-0
$layer_with_weights-1
$layer-1
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
+layer_with_weights-0
+layer-0
,layer_with_weights-1
,layer-1
-layer_with_weights-2
-layer-2
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
4
_variables
5_iterations
6_learning_rate
7_index_dict
8
_momentums
9_velocities
:_update_step_xla"
experimentalOptimizer
,
;serving_default"
signature_map
):'@2conv2d_3/kernel
:@2conv2d_3/bias
):'@ 2conv2d_4/kernel
: 2conv2d_4/bias
3:1  2conv2d_transpose_2/kernel
%:# 2conv2d_transpose_2/bias
3:1@ 2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
):'@2conv2d_5/kernel
:2conv2d_5/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_denoise_layer_call_fn_112526input_1"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
(__inference_denoise_layer_call_fn_112577input_1"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
(__inference_denoise_layer_call_fn_112681x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
(__inference_denoise_layer_call_fn_112706x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
C__inference_denoise_layer_call_and_return_conditional_losses_112448input_1"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
C__inference_denoise_layer_call_and_return_conditional_losses_112474input_1"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
C__inference_denoise_layer_call_and_return_conditional_losses_112771x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
C__inference_denoise_layer_call_and_return_conditional_losses_112836x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

kernel
bias
 C_jit_compiled_convolution_op"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

kernel
bias
 J_jit_compiled_convolution_op"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32�
-__inference_sequential_1_layer_call_fn_112090
-__inference_sequential_1_layer_call_fn_112117
-__inference_sequential_1_layer_call_fn_112849
-__inference_sequential_1_layer_call_fn_112862�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
�
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112048
H__inference_sequential_1_layer_call_and_return_conditional_losses_112062
H__inference_sequential_1_layer_call_and_return_conditional_losses_112880
H__inference_sequential_1_layer_call_and_return_conditional_losses_112898�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

kernel
bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias
 e_jit_compiled_convolution_op"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

kernel
bias
 l_jit_compiled_convolution_op"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_0
strace_1
ttrace_2
utrace_32�
-__inference_sequential_2_layer_call_fn_112341
-__inference_sequential_2_layer_call_fn_112377
-__inference_sequential_2_layer_call_fn_112915
-__inference_sequential_2_layer_call_fn_112932�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0zstrace_1zttrace_2zutrace_3
�
vtrace_0
wtrace_1
xtrace_2
ytrace_32�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112285
H__inference_sequential_2_layer_call_and_return_conditional_losses_112304
H__inference_sequential_2_layer_call_and_return_conditional_losses_112983
H__inference_sequential_2_layer_call_and_return_conditional_losses_113034�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0zwtrace_1zxtrace_2zytrace_3
�
50
z1
{2
|3
}4
~5
6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
m
z0
|1
~2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
m
{0
}1
2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_112656input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_3_layer_call_fn_113043�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_113054�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_4_layer_call_fn_113063�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_113074�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_1_layer_call_fn_112090input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_112117input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_112849inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_112862inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112048input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112062input_2"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112880inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_112898inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_2_layer_call_fn_113083�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_113117�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_3_layer_call_fn_113126�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_113160�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_5_layer_call_fn_113169�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_113180�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_2_layer_call_fn_112341conv2d_transpose_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_112377conv2d_transpose_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_112915inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_112932inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112285conv2d_transpose_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112304conv2d_transpose_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_112983inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_113034inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.:,@2Adam/m/conv2d_3/kernel
.:,@2Adam/v/conv2d_3/kernel
 :@2Adam/m/conv2d_3/bias
 :@2Adam/v/conv2d_3/bias
.:,@ 2Adam/m/conv2d_4/kernel
.:,@ 2Adam/v/conv2d_4/kernel
 : 2Adam/m/conv2d_4/bias
 : 2Adam/v/conv2d_4/bias
8:6  2 Adam/m/conv2d_transpose_2/kernel
8:6  2 Adam/v/conv2d_transpose_2/kernel
*:( 2Adam/m/conv2d_transpose_2/bias
*:( 2Adam/v/conv2d_transpose_2/bias
8:6@ 2 Adam/m/conv2d_transpose_3/kernel
8:6@ 2 Adam/v/conv2d_transpose_3/kernel
*:(@2Adam/m/conv2d_transpose_3/bias
*:(@2Adam/v/conv2d_transpose_3/bias
.:,@2Adam/m/conv2d_5/kernel
.:,@2Adam/v/conv2d_5/kernel
 :2Adam/m/conv2d_5/bias
 :2Adam/v/conv2d_5/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_3_layer_call_fn_113043inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_113054inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_4_layer_call_fn_113063inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_113074inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_2_layer_call_fn_113083inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_113117inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_conv2d_transpose_3_layer_call_fn_113126inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_113160inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_5_layer_call_fn_113169inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_5_layer_call_and_return_conditional_losses_113180inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_112009�
:�7
0�-
+�(
input_1�����������
� "=�:
8
output_1,�)
output_1������������
D__inference_conv2d_3_layer_call_and_return_conditional_losses_113054w9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������@
� �
)__inference_conv2d_3_layer_call_fn_113043l9�6
/�,
*�'
inputs�����������
� "+�(
unknown�����������@�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_113074u9�6
/�,
*�'
inputs�����������@
� "4�1
*�'
tensor_0���������}} 
� �
)__inference_conv2d_4_layer_call_fn_113063j9�6
/�,
*�'
inputs�����������@
� ")�&
unknown���������}} �
D__inference_conv2d_5_layer_call_and_return_conditional_losses_113180w9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������
� �
)__inference_conv2d_5_layer_call_fn_113169l9�6
/�,
*�'
inputs�����������@
� "+�(
unknown������������
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_113117�I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+��������������������������� 
� �
3__inference_conv2d_transpose_2_layer_call_fn_113083�I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+��������������������������� �
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_113160�I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������@
� �
3__inference_conv2d_transpose_3_layer_call_fn_113126�I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+���������������������������@�
C__inference_denoise_layer_call_and_return_conditional_losses_112448�
J�G
0�-
+�(
input_1�����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
C__inference_denoise_layer_call_and_return_conditional_losses_112474�
J�G
0�-
+�(
input_1�����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
C__inference_denoise_layer_call_and_return_conditional_losses_112771�
D�A
*�'
%�"
x�����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
C__inference_denoise_layer_call_and_return_conditional_losses_112836�
D�A
*�'
%�"
x�����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
(__inference_denoise_layer_call_fn_112526�
J�G
0�-
+�(
input_1�����������
�

trainingp"+�(
unknown������������
(__inference_denoise_layer_call_fn_112577�
J�G
0�-
+�(
input_1�����������
�

trainingp "+�(
unknown������������
(__inference_denoise_layer_call_fn_112681
D�A
*�'
%�"
x�����������
�

trainingp"+�(
unknown������������
(__inference_denoise_layer_call_fn_112706
D�A
*�'
%�"
x�����������
�

trainingp "+�(
unknown������������
H__inference_sequential_1_layer_call_and_return_conditional_losses_112048�B�?
8�5
+�(
input_2�����������
p

 
� "4�1
*�'
tensor_0���������}} 
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_112062�B�?
8�5
+�(
input_2�����������
p 

 
� "4�1
*�'
tensor_0���������}} 
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_112880A�>
7�4
*�'
inputs�����������
p

 
� "4�1
*�'
tensor_0���������}} 
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_112898A�>
7�4
*�'
inputs�����������
p 

 
� "4�1
*�'
tensor_0���������}} 
� �
-__inference_sequential_1_layer_call_fn_112090uB�?
8�5
+�(
input_2�����������
p

 
� ")�&
unknown���������}} �
-__inference_sequential_1_layer_call_fn_112117uB�?
8�5
+�(
input_2�����������
p 

 
� ")�&
unknown���������}} �
-__inference_sequential_1_layer_call_fn_112849tA�>
7�4
*�'
inputs�����������
p

 
� ")�&
unknown���������}} �
-__inference_sequential_1_layer_call_fn_112862tA�>
7�4
*�'
inputs�����������
p 

 
� ")�&
unknown���������}} �
H__inference_sequential_2_layer_call_and_return_conditional_losses_112285�Q�N
G�D
:�7
conv2d_transpose_2_input���������}} 
p

 
� "6�3
,�)
tensor_0�����������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_112304�Q�N
G�D
:�7
conv2d_transpose_2_input���������}} 
p 

 
� "6�3
,�)
tensor_0�����������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_112983�?�<
5�2
(�%
inputs���������}} 
p

 
� "6�3
,�)
tensor_0�����������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_113034�?�<
5�2
(�%
inputs���������}} 
p 

 
� "6�3
,�)
tensor_0�����������
� �
-__inference_sequential_2_layer_call_fn_112341�Q�N
G�D
:�7
conv2d_transpose_2_input���������}} 
p

 
� "+�(
unknown������������
-__inference_sequential_2_layer_call_fn_112377�Q�N
G�D
:�7
conv2d_transpose_2_input���������}} 
p 

 
� "+�(
unknown������������
-__inference_sequential_2_layer_call_fn_112915v?�<
5�2
(�%
inputs���������}} 
p

 
� "+�(
unknown������������
-__inference_sequential_2_layer_call_fn_112932v?�<
5�2
(�%
inputs���������}} 
p 

 
� "+�(
unknown������������
$__inference_signature_wrapper_112656�
E�B
� 
;�8
6
input_1+�(
input_1�����������"=�:
8
output_1,�)
output_1�����������
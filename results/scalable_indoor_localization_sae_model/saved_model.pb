¤š
ž˘
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.02unknown8Ň

Adam/sae-hidden-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/sae-hidden-1/kernel/v

.Adam/sae-hidden-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/sae-hidden-0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/sae-hidden-0/kernel/v

.Adam/sae-hidden-0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-0/kernel/v* 
_output_shapes
:
*
dtype0

Adam/sae-hidden-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/sae-hidden-1/kernel/m

.Adam/sae-hidden-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/sae-hidden-0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/sae-hidden-0/kernel/m

.Adam/sae-hidden-0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sae-hidden-0/kernel/m* 
_output_shapes
:
*
dtype0
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

sae-hidden-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namesae-hidden-1/kernel
}
'sae-hidden-1/kernel/Read/ReadVariableOpReadVariableOpsae-hidden-1/kernel* 
_output_shapes
:
*
dtype0

sae-hidden-0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namesae-hidden-0/kernel
}
'sae-hidden-0/kernel/Read/ReadVariableOpReadVariableOpsae-hidden-0/kernel* 
_output_shapes
:
*
dtype0

NoOpNoOp
ŕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel*

0
1*

0
1*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
 trace_1
!trace_2
"trace_3* 
6
#trace_0
$trace_1
%trace_2
&trace_3* 
* 
h
'iter

(beta_1

)beta_2
	*decay
+learning_ratem@mAvBvC*

,serving_default* 

0*

0*
* 

-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

2trace_0* 

3trace_0* 
c]
VARIABLE_VALUEsae-hidden-0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

9trace_0* 

:trace_0* 
c]
VARIABLE_VALUEsae-hidden-1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

;0*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
8
<	variables
=	keras_api
	>total
	?count*

>0
?1*

<	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sae-hidden-0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sae-hidden-1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sae-hidden-0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sae-hidden-1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

"serving_default_sae-hidden-0_inputPlaceholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
î
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sae-hidden-0_inputsae-hidden-0/kernelsae-hidden-1/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_46559
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'sae-hidden-0/kernel/Read/ReadVariableOp'sae-hidden-1/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/sae-hidden-0/kernel/m/Read/ReadVariableOp.Adam/sae-hidden-1/kernel/m/Read/ReadVariableOp.Adam/sae-hidden-0/kernel/v/Read/ReadVariableOp.Adam/sae-hidden-1/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_46693

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesae-hidden-0/kernelsae-hidden-1/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/sae-hidden-0/kernel/mAdam/sae-hidden-1/kernel/mAdam/sae-hidden-0/kernel/vAdam/sae-hidden-1/kernel/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_46742×
Ë
¤
#__inference_signature_wrapper_46559
sae_hidden_0_input
unknown:

	unknown_0:

identity˘StatefulPartitionedCallĹ
StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_46430p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesae-hidden-0_input
Ă
Ă
 __inference__wrapped_model_46430
sae_hidden_0_inputJ
6sequential_sae_hidden_0_matmul_readvariableop_resource:
J
6sequential_sae_hidden_1_matmul_readvariableop_resource:

identity˘-sequential/sae-hidden-0/MatMul/ReadVariableOp˘-sequential/sae-hidden-1/MatMul/ReadVariableOpŚ
-sequential/sae-hidden-0/MatMul/ReadVariableOpReadVariableOp6sequential_sae_hidden_0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
sequential/sae-hidden-0/MatMulMatMulsae_hidden_0_input5sequential/sae-hidden-0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential/sae-hidden-0/ReluRelu(sequential/sae-hidden-0/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
-sequential/sae-hidden-1/MatMul/ReadVariableOpReadVariableOp6sequential_sae_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0ž
sequential/sae-hidden-1/MatMulMatMul*sequential/sae-hidden-0/Relu:activations:05sequential/sae-hidden-1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
sequential/sae-hidden-1/ReluRelu(sequential/sae-hidden-1/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
IdentityIdentity*sequential/sae-hidden-1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
NoOpNoOp.^sequential/sae-hidden-0/MatMul/ReadVariableOp.^sequential/sae-hidden-1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2^
-sequential/sae-hidden-0/MatMul/ReadVariableOp-sequential/sae-hidden-0/MatMul/ReadVariableOp2^
-sequential/sae-hidden-1/MatMul/ReadVariableOp-sequential/sae-hidden-1/MatMul/ReadVariableOp:\ X
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesae-hidden-0_input
Ś
˛
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46445

inputs2
matmul_readvariableop_resource:

identity˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluMatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


E__inference_sequential_layer_call_and_return_conditional_losses_46462

inputs&
sae_hidden_0_46446:
&
sae_hidden_1_46458:

identity˘$sae-hidden-0/StatefulPartitionedCall˘$sae-hidden-1/StatefulPartitionedCallë
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallinputssae_hidden_0_46446*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46445
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_46458*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46457}
IdentityIdentity-sae-hidden-1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
÷
Ť
*__inference_sequential_layer_call_fn_46469
sae_hidden_0_input
unknown:

	unknown_0:

identity˘StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_46462p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesae-hidden-0_input
Ó

*__inference_sequential_layer_call_fn_46568

inputs
unknown:

	unknown_0:

identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_46462p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ś
˛
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46457

inputs2
matmul_readvariableop_resource:

identity˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluMatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
÷
Ť
*__inference_sequential_layer_call_fn_46522
sae_hidden_0_input
unknown:

	unknown_0:

identity˘StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_46506p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesae-hidden-0_input
Ó

*__inference_sequential_layer_call_fn_46577

inputs
unknown:

	unknown_0:

identity˘StatefulPartitionedCallŢ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_46506p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


E__inference_sequential_layer_call_and_return_conditional_losses_46506

inputs&
sae_hidden_0_46499:
&
sae_hidden_1_46502:

identity˘$sae-hidden-0/StatefulPartitionedCall˘$sae-hidden-1/StatefulPartitionedCallë
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallinputssae_hidden_0_46499*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46445
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_46502*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46457}
IdentityIdentity-sae-hidden-1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ś
˛
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46631

inputs2
matmul_readvariableop_resource:

identity˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluMatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ş
°
E__inference_sequential_layer_call_and_return_conditional_losses_46601

inputs?
+sae_hidden_0_matmul_readvariableop_resource:
?
+sae_hidden_1_matmul_readvariableop_resource:

identity˘"sae-hidden-0/MatMul/ReadVariableOp˘"sae-hidden-1/MatMul/ReadVariableOp
"sae-hidden-0/MatMul/ReadVariableOpReadVariableOp+sae_hidden_0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sae-hidden-0/MatMulMatMulinputs*sae-hidden-0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
sae-hidden-0/ReluRelusae-hidden-0/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"sae-hidden-1/MatMul/ReadVariableOpReadVariableOp+sae_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sae-hidden-1/MatMulMatMulsae-hidden-0/Relu:activations:0*sae-hidden-1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
sae-hidden-1/ReluRelusae-hidden-1/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
IdentityIdentitysae-hidden-1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp#^sae-hidden-0/MatMul/ReadVariableOp#^sae-hidden-1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2H
"sae-hidden-0/MatMul/ReadVariableOp"sae-hidden-0/MatMul/ReadVariableOp2H
"sae-hidden-1/MatMul/ReadVariableOp"sae-hidden-1/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Š

,__inference_sae-hidden-0_layer_call_fn_46608

inputs
unknown:

identity˘StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46445p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Š

,__inference_sae-hidden-1_layer_call_fn_46623

inputs
unknown:

identity˘StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46457p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž%
ë
__inference__traced_save_46693
file_prefix2
.savev2_sae_hidden_0_kernel_read_readvariableop2
.savev2_sae_hidden_1_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_sae_hidden_0_kernel_m_read_readvariableop9
5savev2_adam_sae_hidden_1_kernel_m_read_readvariableop9
5savev2_adam_sae_hidden_0_kernel_v_read_readvariableop9
5savev2_adam_sae_hidden_1_kernel_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ä
valueşBˇB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_sae_hidden_0_kernel_read_readvariableop.savev2_sae_hidden_1_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_sae_hidden_0_kernel_m_read_readvariableop5savev2_adam_sae_hidden_1_kernel_m_read_readvariableop5savev2_adam_sae_hidden_0_kernel_v_read_readvariableop5savev2_adam_sae_hidden_1_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*m
_input_shapes\
Z: :
:
: : : : : : : :
:
:
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :&
"
 
_output_shapes
:
:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:

_output_shapes
: 
Ś
˛
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46616

inputs2
matmul_readvariableop_resource:

identity˘MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
ReluReluMatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
7
ý
!__inference__traced_restore_46742
file_prefix8
$assignvariableop_sae_hidden_0_kernel:
:
&assignvariableop_1_sae_hidden_1_kernel:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: A
-assignvariableop_9_adam_sae_hidden_0_kernel_m:
B
.assignvariableop_10_adam_sae_hidden_1_kernel_m:
B
.assignvariableop_11_adam_sae_hidden_0_kernel_v:
B
.assignvariableop_12_adam_sae_hidden_1_kernel_v:

identity_14˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ä
valueşBˇB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_sae_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp&assignvariableop_1_sae_hidden_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp-assignvariableop_9_adam_sae_hidden_0_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp.assignvariableop_10_adam_sae_hidden_1_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp.assignvariableop_11_adam_sae_hidden_0_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp.assignvariableop_12_adam_sae_hidden_1_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 í
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: Ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
ş
°
E__inference_sequential_layer_call_and_return_conditional_losses_46589

inputs?
+sae_hidden_0_matmul_readvariableop_resource:
?
+sae_hidden_1_matmul_readvariableop_resource:

identity˘"sae-hidden-0/MatMul/ReadVariableOp˘"sae-hidden-1/MatMul/ReadVariableOp
"sae-hidden-0/MatMul/ReadVariableOpReadVariableOp+sae_hidden_0_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sae-hidden-0/MatMulMatMulinputs*sae-hidden-0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
sae-hidden-0/ReluRelusae-hidden-0/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"sae-hidden-1/MatMul/ReadVariableOpReadVariableOp+sae_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
sae-hidden-1/MatMulMatMulsae-hidden-0/Relu:activations:0*sae-hidden-1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙k
sae-hidden-1/ReluRelusae-hidden-1/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
IdentityIdentitysae-hidden-1/Relu:activations:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp#^sae-hidden-0/MatMul/ReadVariableOp#^sae-hidden-1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2H
"sae-hidden-0/MatMul/ReadVariableOp"sae-hidden-0/MatMul/ReadVariableOp2H
"sae-hidden-1/MatMul/ReadVariableOp"sae-hidden-1/MatMul/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ś

E__inference_sequential_layer_call_and_return_conditional_losses_46532
sae_hidden_0_input&
sae_hidden_0_46525:
&
sae_hidden_1_46528:

identity˘$sae-hidden-0/StatefulPartitionedCall˘$sae-hidden-1/StatefulPartitionedCall÷
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputsae_hidden_0_46525*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46445
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_46528*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46457}
IdentityIdentity-sae-hidden-1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall:\ X
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesae-hidden-0_input
Ś

E__inference_sequential_layer_call_and_return_conditional_losses_46542
sae_hidden_0_input&
sae_hidden_0_46535:
&
sae_hidden_1_46538:

identity˘$sae-hidden-0/StatefulPartitionedCall˘$sae-hidden-1/StatefulPartitionedCall÷
$sae-hidden-0/StatefulPartitionedCallStatefulPartitionedCallsae_hidden_0_inputsae_hidden_0_46535*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46445
$sae-hidden-1/StatefulPartitionedCallStatefulPartitionedCall-sae-hidden-0/StatefulPartitionedCall:output:0sae_hidden_1_46538*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46457}
IdentityIdentity-sae-hidden-1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp%^sae-hidden-0/StatefulPartitionedCall%^sae-hidden-1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 2L
$sae-hidden-0/StatefulPartitionedCall$sae-hidden-0/StatefulPartitionedCall2L
$sae-hidden-1/StatefulPartitionedCall$sae-hidden-1/StatefulPartitionedCall:\ X
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesae-hidden-0_input"żL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_defaultł
R
sae-hidden-0_input<
$serving_default_sae-hidden-0_input:0˙˙˙˙˙˙˙˙˙A
sae-hidden-11
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:âW
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
ą
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel"
_tf_keras_layer
ą
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ţ
trace_0
 trace_1
!trace_2
"trace_32ó
*__inference_sequential_layer_call_fn_46469
*__inference_sequential_layer_call_fn_46568
*__inference_sequential_layer_call_fn_46577
*__inference_sequential_layer_call_fn_46522Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 ztrace_0z trace_1z!trace_2z"trace_3
Ę
#trace_0
$trace_1
%trace_2
&trace_32ß
E__inference_sequential_layer_call_and_return_conditional_losses_46589
E__inference_sequential_layer_call_and_return_conditional_losses_46601
E__inference_sequential_layer_call_and_return_conditional_losses_46532
E__inference_sequential_layer_call_and_return_conditional_losses_46542Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z#trace_0z$trace_1z%trace_2z&trace_3
ÖBÓ
 __inference__wrapped_model_46430sae-hidden-0_input"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
w
'iter

(beta_1

)beta_2
	*decay
+learning_ratem@mAvBvC"
	optimizer
,
,serving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
đ
2trace_02Ó
,__inference_sae-hidden-0_layer_call_fn_46608˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z2trace_0

3trace_02î
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46616˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z3trace_0
':%
2sae-hidden-0/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
đ
9trace_02Ó
,__inference_sae-hidden-1_layer_call_fn_46623˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z9trace_0

:trace_02î
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46631˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z:trace_0
':%
2sae-hidden-1/kernel
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_sequential_layer_call_fn_46469sae-hidden-0_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
üBů
*__inference_sequential_layer_call_fn_46568inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
üBů
*__inference_sequential_layer_call_fn_46577inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
*__inference_sequential_layer_call_fn_46522sae-hidden-0_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_46589inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_46601inputs"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŁB 
E__inference_sequential_layer_call_and_return_conditional_losses_46532sae-hidden-0_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ŁB 
E__inference_sequential_layer_call_and_return_conditional_losses_46542sae-hidden-0_input"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ŐBŇ
#__inference_signature_wrapper_46559sae-hidden-0_input"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ŕBÝ
,__inference_sae-hidden-0_layer_call_fn_46608inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46616inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
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
ŕBÝ
,__inference_sae-hidden-1_layer_call_fn_46623inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46631inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
N
<	variables
=	keras_api
	>total
	?count"
_tf_keras_metric
.
>0
?1"
trackable_list_wrapper
-
<	variables"
_generic_user_object
:  (2total
:  (2count
,:*
2Adam/sae-hidden-0/kernel/m
,:*
2Adam/sae-hidden-1/kernel/m
,:*
2Adam/sae-hidden-0/kernel/v
,:*
2Adam/sae-hidden-1/kernel/vĽ
 __inference__wrapped_model_46430<˘9
2˘/
-*
sae-hidden-0_input˙˙˙˙˙˙˙˙˙
Ş "<Ş9
7
sae-hidden-1'$
sae-hidden-1˙˙˙˙˙˙˙˙˙¨
G__inference_sae-hidden-0_layer_call_and_return_conditional_losses_46616]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_sae-hidden-0_layer_call_fn_46608P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙¨
G__inference_sae-hidden-1_layer_call_and_return_conditional_losses_46631]0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_sae-hidden-1_layer_call_fn_46623P0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ť
E__inference_sequential_layer_call_and_return_conditional_losses_46532rD˘A
:˘7
-*
sae-hidden-0_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ť
E__inference_sequential_layer_call_and_return_conditional_losses_46542rD˘A
:˘7
-*
sae-hidden-0_input˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ż
E__inference_sequential_layer_call_and_return_conditional_losses_46589f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ż
E__inference_sequential_layer_call_and_return_conditional_losses_46601f8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
*__inference_sequential_layer_call_fn_46469eD˘A
:˘7
-*
sae-hidden-0_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
*__inference_sequential_layer_call_fn_46522eD˘A
:˘7
-*
sae-hidden-0_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
*__inference_sequential_layer_call_fn_46568Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
*__inference_sequential_layer_call_fn_46577Y8˘5
.˘+
!
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙ž
#__inference_signature_wrapper_46559R˘O
˘ 
HŞE
C
sae-hidden-0_input-*
sae-hidden-0_input˙˙˙˙˙˙˙˙˙"<Ş9
7
sae-hidden-1'$
sae-hidden-1˙˙˙˙˙˙˙˙˙
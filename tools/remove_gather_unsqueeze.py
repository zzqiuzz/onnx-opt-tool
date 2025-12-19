import onnx    
# 加载现有的ONNX模型
model = onnx.load("/home/uto/workspace/demos/bevod/online_const_ogs.onnx")
onnx_graph_ = model.graph
all_node = model.graph.node

rep_cnt_ = 0 

node_list_removed = []

for i, node in enumerate(all_node):
    if node.name in ['Gather_160','Unsqueeze_242']:
        node_list_removed.append(node)
    
    if node.name == 'Transpose_243':
        node.input[0] = '859'
        
        
        # 删除要被替换的层
for rm_node in node_list_removed:
    print('@@@ this time rm_node = ', rm_node.name)
    onnx_graph_.node.remove(rm_node)

# 保存修改后的模型
onnx.save(model, "/home/uto/workspace/demos/bevod/opt/opt.onnx")
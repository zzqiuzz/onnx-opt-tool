import onnx

def find_father_node_of_a_tensor(tensor_name, all_node):
    father_node = ''
    for node in all_node:
        if tensor_name in node.output:
            father_node = node
            break
    return father_node


# 加载现有的ONNX模型
model = onnx.load(args_model_path)
onnx_graph_ = model.graph
all_node = model.graph.node

rep_cnt_ = 0 
for i, node in enumerate(all_node):
    if 'Log' in node.name:
        
        print('****************************** iteration start *****************************')
        print('@@@@@@ node ',i,': ', node.name, ', out = ', node.output[0], ', input = ', node.input[0])

        # 从log出发
        this_log_ly_name = node.name
        this_log_in_tensor = node.input[0]
        this_log_out_tensor = node.output
        
        # 本次迭代需要被remove的所有node list
        node_list_removed = []
        # log节点本身需要被remove,所以加入removed list
        node_list_removed.append(node)####

        #0000 往上寻找Div层，如果不是则continue
        father_node0 = find_father_node_of_a_tensor(this_log_in_tensor, all_node)
        if 'Div' not in father_node0.name:
            continue
        
        #0000 如果是的话, Div也需要干掉
        node_list_removed.append(father_node0)####
        
        # 除数和被除数
        div_inputA_name = father_node0.input[0]
        div_inputB_name = father_node0.input[1]
        
        # 创建一个新的Log节点log_node_A
        log_node_A = onnx.helper.make_node(
            op_type='Log',
            inputs=[div_inputA_name],
            outputs=['log_A_out_'+str(rep_cnt_)],
            name='Log_A_'+str(rep_cnt_)
        )
        
        # 创建一个新的Log节点log_node_B
        log_node_B = onnx.helper.make_node(
            op_type='Log',
            inputs=[div_inputB_name],
            outputs=['log_B_out_'+str(rep_cnt_)],
            name='Log_B_'+str(rep_cnt_)
        )
        
        # 创建一个新的Sub节点Sub_node
        Sub_node = onnx.helper.make_node(
            op_type='Sub',
            inputs=['log_A_out_'+str(rep_cnt_), 'log_B_out_'+str(rep_cnt_)],
            outputs=this_log_out_tensor,
            name='Sub_logA_logB_'+str(rep_cnt_)
        )
        rep_cnt_+=1
        
        # 增加到模型里面
        onnx_graph_.node.append(log_node_A)
        onnx_graph_.node.append(log_node_B)
        onnx_graph_.node.append(Sub_node)
        
        # 删除要被替换的层
        for rm_node in node_list_removed:
            print('@@@ this time rm_node = ', rm_node.name)
            onnx_graph_.node.remove(rm_node)
        
        print('****************************** iteration end ***************************** rep_cnt_ = ',rep_cnt_)

    # 保存修改后的模型
    onnx.save(model, args_output_path)
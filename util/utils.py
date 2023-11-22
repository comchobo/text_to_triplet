def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def flatten_and_batch_dict(original_list):
    batch_dict = {n:len(x) for n, x in enumerate(original_list)}
    temp_lists = [item for sublist in original_list for item in sublist]
    return batch_dict, temp_lists

def return_batch_to_list(batch_dict, batched_list):
    original_list = []
    counter = 0
    for key in batch_dict.keys():
        temp = []
        for i in range(batch_dict[key]):
            temp.append(batched_list[counter])
            counter+=1
        original_list.append(temp)
    return original_list
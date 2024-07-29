def print_parsed_result(d):
    res = d['node']
    rule = d['rule']
    
    if rule == 1:
        res += '\n\t(\n'
        res += '\t\t' + print_parsed_result(d['mid']).replace('\n', '\n\t\t') + '\n'
        res += '\t)'
    elif rule == 2:
        for n in d['nodes']:
            res += '\n\t' + print_parsed_result(n).replace('\n', '\n\t')
    
    return res 
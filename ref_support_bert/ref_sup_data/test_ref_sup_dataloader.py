from ref_support_bert.ref_sup_data.ref_sup_dataloader import get_ref_sup_dataloader

if __name__ == '__main__':
    ref_sup_dataloader = get_ref_sup_dataloader()
    for data in ref_sup_dataloader:
        print(data)

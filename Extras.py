def ImputeInBlocks(data,itemsLimit=10000):
    
    [N,m] = data.shape
    imp = Imputer(missing_values="NaN", strategy='most_frequent', axis=0,verbose=10)
    count = 0
    while count<N:
        temp = data[count:count+itemsLimit,:]
        vals,vals_freq = mode(temp)
        data[np.isnan[data.data[data.indptr[count]:data.indptr[count+itemsLimit]]]] = vals
        imp.fit(temp)
        data.data[data.indptr[count]:data.indptr[count+itemsLimit],:] = imp.transform(temp)
        count += itemsLimit
        logging.debug("Imputing "+str(count)+" items done")
        
    temp = data[count-itemsLimit:N,:]
    imp.fit(temp)
    data.data[count-itemsLimit:N,:] = imp.transform(temp)
        
    return data

def RunCatConversion(data,enc=None):
    [N,m] = data.shape
    
    if not enc:
        ncols = 0
        for i in range(m):
            ncols += len(np.unique(data[:,i]))
    else:
        ncols = 0
        for keys in enc.keys():
            ncols += len(enc[keys])
            
    enc_out = {}
    rows=[]
    cols=[]
    if not enc:
        col=0
        for i in range(m):
            logging.debug("Starting column " + str(i))        
            un_vals = np.unique(data[:,i])
            for e,vals in enumerate(un_vals):
                locs = np.where(data[:,i]==e)
                rows.append(locs)
                cols.append(col*np.ones(len(locs)))
                col += 1
            enc_out[i] = un_vals
#            temp = np.array([data[:,i]])
#            data_bin[:,col:col+len(un_vals)] = (temp.T==un_vals).astype('int')
#            col += len(un_vals)
    else:
        col = 0
        for i in range(m):
            un_vals = enc[i]
            for e,vals in enumerate(un_vals):
                locs = np.where(data[:,i]==e)
                rows.append(locs)
                cols.append(col*np.ones(len(locs)))
                col += 1            
#            temp = np.array([data[:,i]])
#            data_bin[:,col:col+len(un_vals)] = (temp.T==un_vals).astype('int')
#            col += len(un_vals)
    data = sp.csr_matrix((np.ones(len(rows)),(rows,cols)), shape=(N, m), dtype=np.float64)

    if enc_out:
        return data, enc_out
    else:
        return data

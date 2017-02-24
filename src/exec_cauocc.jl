using ArgParse
using Cauocc
using Cauocc.Misc: dict_to_adjmat

function get_settings()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--test_name", "-t"
        help = "short form of test to be used"
        arg_type = String
        default = "mi"
        "--max_k", "-m"
        help = "maximum size of conditioning sets"
        arg_type = Int
        default = 3
        "--alpha", "-a"
        help = "significance niveau for tests"
        arg_type = Float64
        default = 0.01
        "--pwr", "-p"
        help = "minimum fraction of tests with sufficient power"
        arg_type = Float64
        default = 0.5
        "--weight_type", "-w"
        help = "type of edge weights to be used"
        arg_type = String
        default = "cond_logpval"
        "--debug", "-d"
        help = "debug level"
        arg_type = Int
        default = 0
        "--FDR", "-f"
        help = "do FDR correction"
        action = :store_true
        "--header"
        help = "does table have a header"
        action = :store_true
        "--rownames"
        help = "does table have a row names"
        action = :store_true
        "in_path"
        help = "input table path"
        required = true
        "out_path"
        help = "output table path"
        required = true    
    end
    
    parsed_args = parse_args(ARGS, s)
end

function main()
    parsed_args = get_settings()
    
    data = readdlm(parsed_args["in_path"], '\t')
    
    if parsed_args["header"]
        header = convert(Vector{String}, data[1, :])
        data = data[2:end, :]
    else
        header = String[]
    end
    
    if parsed_args["rownames"]
        data = data[:, 2:end]
        
        if !isempty(header)
            header = header[2:end]
        end
    end
    
    if parsed_args["test_name"] in ["mi", "mi_nz"]
        data = convert(Matrix{Int}, data)
    else
        data = convert(Matrix{Float64}, data)
    end
    
    #if parsed_args["test_name"] == "mi"
    #    data = sign(data)
    #elseif parsed_args["test_name"] == "mi_nz"
    #end
    
    if nprocs() > 1
        parallel = "multi_ep"
    else
        parallel = "single"
    end
        
    # have to import here in order to also get module on the workers
    

        
    PC_dict = LGL(data, test_name=parsed_args["test_name"], max_k=parsed_args["max_k"], alpha=parsed_args["alpha"], 
    pwr=parsed_args["pwr"], parallel=parallel, FDR=parsed_args["FDR"], weight_type=parsed_args["weight_type"], header=header, debug=parsed_args["debug"])
    #println(length(PC_dict), " ", sum(map(length, values(PC_dict))))
    
    adj_matrix = dict_to_adjmat(PC_dict)
    writedlm(parsed_args["out_path"], adj_matrix, '\t')
    #open(parsed_args["out_path"], "w") do f
    #    serialize(f, PC_dict)
    #end
    
end
        
main()

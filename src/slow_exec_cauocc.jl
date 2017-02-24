using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--opt1"
            help = "an option with an argument"
        "--opt2", "-o"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--flag1"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "arg1"
            help = "a positional argument"
            required = true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
end

@time main()

"""
using ArgParse
#using Cauocc.Learning: LGL

function get_settings_macro()
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
    
    #parsed_args = parse_args(ARGS, s)
end

function get_settings()
    
end

function main()
    parsed_args = get_settings_macro()
end

@time main()
"""
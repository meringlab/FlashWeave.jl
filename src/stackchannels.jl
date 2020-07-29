import Base: AbstractChannel, put!, take!, isready, isopen, close, lock, unlock, check_channel_state, notify

# not ideal but Base doesn't leave us much choice
function stacktake!(c::Channel)
    lock(c)
    try
        while isempty(c.data)
            check_channel_state(c)
            wait(c.cond_take)
        end
        v = pop!(c.data) # only line changed from Base.take!()
        notify(c.cond_put, nothing, false, false)
        return v
    finally
        unlock(c)
    end
end

struct StackChannel{T} <: AbstractChannel{T}
    channel::Channel{T}
end

StackChannel{T}(n::Int) where T = StackChannel(Channel{T}(n))

put!(lc::StackChannel, x) = put!(lc.channel, x)
take!(lc::StackChannel) = stacktake!(lc.channel)

close(lc::StackChannel) = close(lc.channel)
isopen(lc::StackChannel) = isopen(lc.channel)

isready(lc::StackChannel) = isready(lc.channel)

lock(lc::StackChannel) = lock(lc.channel)
unlock(lc::StackChannel) = unlock(lc.channel)

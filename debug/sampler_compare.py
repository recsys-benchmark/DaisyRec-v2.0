from daisy.utils.sampler import BasicNegtiveSampler
import time
from sys import exit

def sampler_compare(train_set, config):

    for num_neg in 4, 5, 6, 7, 8, 9:
        print("\n======================================")
        print(f'Now calculating for {num_neg} negative samples per user-item pair:')

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.sampling()
        e = time.time()
        old_time = e-s
        print(f'Prev sampling time: {round(old_time, 5)}')
        assert len(sampler.df) == len(train_set) * num_neg
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.guess_and_check_sampling()
        e = time.time()
        guessandcheck_time = e-s
        print(f'Guess and check sampling time: {round(guessandcheck_time,3)}s')
        assert len(sampler.df) == len(train_set) * num_neg
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.set_diff_sampling()
        e = time.time()
        setdifftime = e-s
        print(f'Set diff sampling time: {round(setdifftime, 4)}')
        assert len(sampler.df) == len(train_set) * num_neg
        del sampler


        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.batch_sampling(sampling_batch_size=64)
        e = time.time()
        new_64_time = e-s
        print(f'Batch-aware sampling time: {round(new_64_time,3)}s (batch-size=64)')
        assert len(sampler.df) == len(train_set) * num_neg
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.batch_sampling(sampling_batch_size=128)
        e = time.time()
        new_128_time = e-s
        print(f'Batch-aware sampling time: {round(new_128_time,3)}s (batch-size=128)')
        assert len(sampler.df) == len(train_set) * num_neg
        del sampler

        sampler = BasicNegtiveSampler(train_set.copy(), config)
        sampler.num_ng = num_neg
        s = time.time()
        sampler.batch_sampling(sampling_batch_size=256)
        e = time.time()
        new_256_time = e-s
        print(f'Batch-aware sampling time: {round(new_256_time,3)}s  (batch-size=256)')
        assert len(sampler.df) == len(train_set) * num_neg
        del sampler

        print(f'\nFor {num_neg} negative items per user-item pair: ')
        print(f'The guess-and-check sampling time is {round(guessandcheck_time/old_time, 2)} times slower')
        print(f'The set difference  sampling time is {round(setdifftime/old_time, 2)} times slower')
        print(f'The  64 batch-aware sampling time is {round(new_64_time/old_time, 2)} times slower')
        print(f'The 128 batch-aware sampling time is {round(new_128_time/old_time, 2)} times slower')
        print(f'The 256 batch-aware sampling time is {round(new_256_time/old_time, 2)} times slower')

    exit()

def sampler_compare2(train_set, config):

    sampler = BasicNegtiveSampler(train_set.copy(), config)
    sampler.num_ng = 4
    s = time.time()
    sampler.sampling()
    e = time.time()
    old_time = e-s
    print(f'Prev sampling time: {round(old_time, 5)}')
    assert len(sampler.df) == len(train_set) * 4
    del sampler

    sampler = BasicNegtiveSampler(train_set.copy(), config)
    sampler.num_ng = 4
    s = time.time()
    sampler.guess_and_check_sampling()
    e = time.time()
    guessandcheck_time = e-s
    print(f'Guess and check sampling time: {round(guessandcheck_time,3)}s')
    assert len(sampler.df) == len(train_set) * 4
    del sampler

    print(f'The guess-and-check sampling time is {round(guessandcheck_time/old_time, 2)} times slower')

    exit()
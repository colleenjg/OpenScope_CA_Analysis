import analyse_pupil as analyse_pupil


if __name__ == "__main__":
    
    sesses = [758519303, 761624763, 760260459, 761730740, 764704289]
    runtypes = ['prod', 'prod', 'prod', 'prod', 'prod']

    for stimtype in ['gabors', 'bricks']:
        for sess, runtype in zip(sesses[3:], runtypes[3:]):
            analyse_pupil.quick_run(sess, stimtype, runtype)

    
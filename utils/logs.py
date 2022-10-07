def log_string(log, string, pnt=True):  # p decide print
    log.write(string + '\n')
    log.flush()
    if pnt:
        print(string)
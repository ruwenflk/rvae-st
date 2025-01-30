def log(message):
    print(message)
    return message


def log_headline(message):
    return log("{} {} {}".format(35 * "#", message, 35 * "#"))


def log_separator():
    return log("{}".format(70 * "#"))

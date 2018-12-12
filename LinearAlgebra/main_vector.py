from getLA.Vector import Vector

if __name__ == "__main__":
    vec = Vector([5, 2])
    print(vec)
    print(len(vec))
    print("vec[0] = {}, vec[1] = {}".format(vec[0], vec[1]))

    vec2 = Vector([3, 1])
    print("{} + {} = {}".format(vec, vec2, vec + vec2))
    print("{} - {} = {}".format(vec, vec2, vec - vec2))

    print("{} * {} = {}".format(vec, 5, vec * 5))
    print("{} * {} = {}".format(5, vec, 5 * vec))

    print("+{} = {}".format(vec, +vec))
    print("-{} = {}".format(vec, -vec))

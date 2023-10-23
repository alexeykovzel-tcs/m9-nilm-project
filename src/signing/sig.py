import lph


def closest_idx(arr, el):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - el))


def sign_signal(signal):
    hist = lph.histogram(signal, 40)
    return hist / hist.sum()


def trunc_tag_pf(data, tag):
    _, _, start, end = tag
    start_idx = closest_idx(data['TimeTicks'], start)
    end_idx = closest_idx(data['TimeTicks'], end)
    return data['Pf'][start_idx:end_idx]


def sign_tag(data, tag):
    signal = trunc_tag_pf(data, tag)
    return sign_signal(signal)


def sign_and_plot_tags(data):
    tags = data['TaggingInfo']
    tag_names = [tag[1] for tag in tags]
    hists = [sign_tag(data, tag) for tag in tags]
    lph.plot_hists(tag_names, hists)

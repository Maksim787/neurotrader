import tinkoff.invest as inv


def quotation_to_float(amount: inv.Quotation) -> float:
    return amount.units + amount.nano / 1e9

import tinkoff.invest as inv


def quotation_to_float(amount: inv.Quotation | inv.MoneyValue) -> float:
    return amount.units + amount.nano / 1e9


def float_to_quotation(amount: float) -> inv.Quotation:
    mult = 10 ** 9
    nano = round(amount * mult)
    return inv.Quotation(units=nano // mult, nano=nano % mult)

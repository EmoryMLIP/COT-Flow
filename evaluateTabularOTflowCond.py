import argparse
from torch.utils.data import DataLoader
from lib.tabloader import tabloader
import lib.utils as utils
from src.OTFlowProblem import *
from src.mmd import mmd
import config

cf = config.getconfig()

parser = argparse.ArgumentParser('COT-Flow')
parser.add_argument(
    '--data', choices=['concrete', 'energy', 'yacht'], type=str, default='concrete'
)
parser.add_argument('--resume', type=str, default="experiments/cnf/tabcond/...")

parser.add_argument('--prec', type=str, default='single', choices=['None', 'single','double'], help="overwrite trained precision")
parser.add_argument('--gpu' , type=int, default=0)
parser.add_argument('--long_version'  , action='store_true')
# default is: args.long_version=False , passing  --long_version will take a long time to run to get values for paper
args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


def compute_loss(net, x,y, nt):
    Jc , cs = OTFlowProblem(x, y, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    print(checkpt['args'])

    data = checkpt['args'].data
    test_ratio = checkpt['args'].test_ratio
    valid_ratio = checkpt['args'].valid_ratio
    batch_size = checkpt['args'].batch_size
    random_state = checkpt['args'].random_state

    _, _, testData, _ = tabloader(data, batch_size, test_ratio, valid_ratio, random_state)
    testLoader = DataLoader(
        testData,
        batch_size=batch_size, shuffle=True
    )
    nex = testData.shape[0]
    d   = testData.shape[1]
    dx = checkpt['args'].dx
    dy = d - dx
    nt_test = checkpt['args'].nt_val

    # reload model
    m       = checkpt['args'].m
    alph    = checkpt['args'].alph
    nTh     = checkpt['args'].nTh

    argPrec = checkpt['state_dict_x']['A'].dtype
    net_x = Phi(nTh=nTh, m=m, dx=dx, dy=dy, alph=alph)
    net_x = net_x.to(argPrec)
    net_x.load_state_dict(checkpt["state_dict_x"])
    net_x = net_x.to(device)

    # if specified precision supplied, override the loaded precision
    if args.prec != 'None':
        if args.prec == 'single':
            argPrec = torch.float32
        if args.prec == 'double':
            argPrec = torch.float64
        net_x = net_x.to(argPrec)

    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    net_x.eval()

    with torch.no_grad():

        # meters to hold testing results
        testLossMeter  = utils.AverageMeter()
        testAlphMeterL = utils.AverageMeter()
        testAlphMeterC = utils.AverageMeter()
        testAlphMeterR = utils.AverageMeter()

        itr = 1
        for x0 in testLoader:
            x0 = cvt(x0)
            nex_batch = x0.shape[0]
            x_test = x0[:, dy:].view(-1, dx)
            y_test = x0[:, :dy].view(-1, dy)
            tst_loss_x, tst_costs_x = compute_loss(net_x, x_test, y_test, nt=nt_test)

            testLossMeter.update(tst_loss_x.item(), nex_batch)
            testAlphMeterL.update(tst_costs_x[0].item(), nex_batch)
            testAlphMeterC.update(tst_costs_x[1].item(), nex_batch)
            testAlphMeterR.update(tst_costs_x[2].item(), nex_batch)

        print('Mean Negative Log Likelihood: {:.3e}'.format(testAlphMeterC.avg))

        # generate samples
        normSamples = torch.randn(testData.shape[0], 1).to(device)

        zx = cvt(normSamples)
        finvx = integrate(zx, testData[:, :dy].to(device), net_x, [1.0, 0.0], nt_test, stepper="rk4", alph=net_x.alph)
        modelGen = finvx[:, :dx].detach().cpu().numpy()
        # compute MMD
        print('Maximum Mean Discrepancy: {:.3e}'.format(mmd(modelGen, testData[:, dy:])))

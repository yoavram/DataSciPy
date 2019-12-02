def generalized_logistic(t, N0, K, r, ν):
    return K / (1 - (1 - K/N0) * np.exp(-ν * r * t))**(1/ν)

plt.plot(t, N, '.k')

params, cov = curve_fit(logistic, t, N, (N.min(), N.max(), 1))
Nhat = logistic(t, *params)
print('Logistic:')
print('\tN0={:.3f}, K={:.3f}, r={:.3f}'.format(*params))
rss0 = ((N - Nhat)**2).sum()
print("\tRSS: {:4f}".format(rss0))
plt.plot(t, Nhat, label='Logistic')

params, cov = curve_fit(generalized_logistic, t, N, (N.min(), N.max(), 1, 1))
Nhat = generalized_logistic(t, *params)
print('Generalized logistic:')
print('\tN0={:.3f}, K={:.3f}, r={:.3f}, ν={:.3f}'.format(*params))
rss1 = ((N - Nhat)**2).mean()
print("\tRSS: {:4f}".format(rss1))
plt.plot(t, Nhat, label='Generalized')

plt.xlabel('Time')
plt.ylabel('OD')
plt.legend(loc='lower right')
sns.despine();
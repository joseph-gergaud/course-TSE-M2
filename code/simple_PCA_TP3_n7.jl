First_Data =[5.1 3.5 1.4 0.2
         4.9 3.0 1.4 0.2
         4.7 3.2 1.3 0.2
         4.6 3.1 1.5 0.2
         5.0 3.6 1.4 0.2
         7.0 3.2 4.7 1.4
         6.4 3.2 4.5 1.5
         6.9 3.1 4.9 1.5
         5.5 2.3 4.0 1.3
         6.5 2.8 4.6 1.5
         6.3 3.3 6.0 2.5
         5.8 2.7 5.1 1.9
         7.1 3.0 5.9 2.1
         6.3 2.9 5.6 1.8
         6.5 3.0 5.8 2.2 ];
X = First_Data[:,[1,2,4]]

p1 = scatter(X[:,1],X[:,2],X[:,3],xlabel = Names[1],ylabel=Names[2],zlabel=Names[3],legend=false)
p2 = scatter(X[:,1],X[:,2],xlabel = Names[1],ylabel=Names[2],legend=false)
p3 = scatter(X[:,1],X[:,3],xlabel = Names[1],ylabel=Names[3],legend=false)
plot(p1,p2,p3,layout=@layout([a ; b c]))